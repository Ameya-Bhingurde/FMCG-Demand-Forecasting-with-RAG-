#!/usr/bin/env python3
"""
FMCG RAG Pipeline
FAISS retrieval + LangChain orchestration + Llama 3 via Groq for generative answers.
Pandas-based analytical answers handle numeric questions for speed and accuracy.
"""

import os
import logging
import pandas as pd
from pathlib import Path
import json
import yaml
import matplotlib.pyplot as plt
import io
import base64
import pickle
import faiss

# Load environment variables from .env (resolved relative to this module, not CWD)
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH)
except ImportError:
    pass

# Embeddings
from sentence_transformers import SentenceTransformer

# LangChain + Groq for generative RAG
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FMCGRAGPipeline")

class FMCGRAGPipeline:
    def __init__(self, config_path=None):
        """Initialize pipeline"""
        # Determine project root and config path
        if config_path is None:
            # Try to find config.yaml in current dir or parent dirs
            current = Path.cwd()
            config_path = None
            
            # Search up to 3 levels
            for _ in range(3):
                candidate = current / "config.yaml"
                if candidate.exists():
                    config_path = candidate
                    break
                current = current.parent
            
            if config_path is None:
                raise FileNotFoundError(
                    "config.yaml not found. Please run from project root or provide config_path"
                )
        
        # Load config
        config_path = Path(config_path)
        project_root = config_path.parent
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.rag_config = self.config.get("rag", {})

        # Paths (absolute, relative to project root)
        self.data_dir = project_root / "data"
        self.vector_store_dir = project_root / "vector_store"
        self.reports_dir = project_root / "reports"
        self.vector_store_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        logger.info(f"RAG Pipeline initialized with project root: {project_root}")

        # Embedding model
        model_name = self.rag_config.get("embedding_model",
                                         "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

        # LangChain + Groq generative LLM
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            logger.warning(
                "GROQ_API_KEY not set. Generative RAG fallback will fail. "
                "Set it in your .env file (see .env.example)."
            )
        self.llm = ChatGroq(
            api_key=groq_api_key or "missing",
            model_name=self.rag_config.get("groq_model", "llama-3.1-8b-instant"),
            temperature=self.rag_config.get("temperature", 0.2),
            max_tokens=self.rag_config.get("max_tokens", 512),
            max_retries=5,            # survive transient network blips
            timeout=30,               # don't hang the UI on a slow request
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior FMCG sales analyst answering questions for a business "
             "stakeholder. Use ONLY the context below — never invent data that isn't "
             "there. Respect every filter the user mentions (product, region, year, "
             "category, brand): if the user asks about 'milk in PL-South in 2023', "
             "only use context lines that match those filters and explicitly say so. "
             "If the context lacks the information to answer accurately, say so "
             "honestly and describe what you DO see in the retrieved context.\n\n"
             "Style rules:\n"
             "- Provide a thorough, well-structured answer of 4-7 sentences.\n"
             "- Be quantitative — cite specific numbers from the context.\n"
             "- For comparison or causation questions, address both sides explicitly.\n"
             "- End with a one-line takeaway when natural.\n\n"
             "=== CONTEXT ===\n{context}"),
            ("human", "{question}")
        ])
        self.qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        logger.info(f"Initialized LangChain Groq chain: model={self.llm.model_name}")

        # Vector store
        self.vector_store = None
        self.documents = []
        self.embeddings = None

    # ===== Legacy methods for compatibility =====
    def setup_embeddings(self):
        logger.info("setup_embeddings() called — embeddings already initialized in __init__.")

    def setup_llm(self):
        logger.info("setup_llm() called — LangChain Groq chain already initialized in __init__.")

    # =============== Data ===============
    def load_sales_data(self):
        path = self.data_dir / "processed" / "cleaned.csv"
        if not path.exists():
            raise FileNotFoundError(f"No sales data at {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # =============== Documents ===============
    def create_documents(self, df):
        """
        Build a multi-resolution summary corpus so the RAG retriever has the
        right grain of pre-aggregated context regardless of how a question is
        framed. Roll-ups are placed first so they tend to win retrieval for
        aggregate questions; daily/SKU/region facts come after for granular
        questions.
        """
        df = df.copy()
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['month'] = pd.DatetimeIndex(df['date']).month
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                       9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        docs = []

        # === Roll-up summaries (~70 docs) ===

        # Overall total
        total_units = df['units_sold'].sum()
        avg_price = df['price_unit'].mean()
        total_revenue = (df['units_sold'] * df['price_unit']).sum()
        years_present = sorted(df['year'].unique().tolist())
        docs.append({
            "content": (
                f"Overall total sales across the full dataset "
                f"({years_present[0]}-{years_present[-1]}): {total_units:,.0f} units sold "
                f"across {df['region'].nunique()} regions, {df['sku'].nunique()} SKUs, "
                f"and {df['category'].nunique()} categories. "
                f"Average price per unit: ${avg_price:.2f}. "
                f"Total revenue: ${total_revenue:,.0f}."
            ),
            "metadata": {"scope": "overall"},
        })

        # Per-year totals — opening phrased to match natural questions
        for year, g in df.groupby('year'):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            rev = (g['units_sold'] * g['price_unit']).sum()
            promos = int((g['promotion_flag'] == 1).sum())
            docs.append({
                "content": (
                    f"Total sales in {year} were {units:,.0f} units. "
                    f"Average price was ${ap:.2f} per unit, total revenue ${rev:,.0f}. "
                    f"Promotion-flagged transactions: {promos:,}."
                ),
                "metadata": {"scope": "year", "year": int(year)},
            })

        # Per-year, per-region
        for (year, region), g in df.groupby(['year', 'region']):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            docs.append({
                "content": (
                    f"Total sales in region {region} in {year} were {units:,.0f} units "
                    f"at average price ${ap:.2f} per unit."
                ),
                "metadata": {"scope": "year_region", "year": int(year), "region": region},
            })

        # Per-year, per-category
        for (year, cat), g in df.groupby(['year', 'category']):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            docs.append({
                "content": (
                    f"Total sales of {cat} category in {year} were {units:,.0f} units "
                    f"at average price ${ap:.2f} per unit."
                ),
                "metadata": {"scope": "year_category", "year": int(year), "category": cat},
            })

        # Per-year, per-region, per-category (key cross-filter)
        for (year, region, cat), g in df.groupby(['year', 'region', 'category']):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            docs.append({
                "content": (
                    f"Total sales of {cat} category in region {region} in {year} were "
                    f"{units:,.0f} units at average price ${ap:.2f} per unit."
                ),
                "metadata": {
                    "scope": "year_region_category",
                    "year": int(year),
                    "region": region,
                    "category": cat,
                },
            })

        # Per-year, per-month (handles "which month had highest sales")
        for (year, month), g in df.groupby(['year', 'month']):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            docs.append({
                "content": (
                    f"Total sales in {month_names[int(month)]} {year} were "
                    f"{units:,.0f} units at average price ${ap:.2f} per unit."
                ),
                "metadata": {
                    "scope": "year_month",
                    "year": int(year),
                    "month": int(month),
                    "month_name": month_names[int(month)],
                },
            })

        # Per-year, per-month, per-category (handles "best month for milk in 2023")
        for (year, month, cat), g in df.groupby(['year', 'month', 'category']):
            units = g['units_sold'].sum()
            ap = g['price_unit'].mean()
            docs.append({
                "content": (
                    f"Total sales of {cat} category in {month_names[int(month)]} {year} "
                    f"were {units:,.0f} units at average price ${ap:.2f} per unit."
                ),
                "metadata": {
                    "scope": "year_month_category",
                    "year": int(year),
                    "month": int(month),
                    "category": cat,
                },
            })

        # Per-year monthly breakdown — single doc listing all 12 months with
        # the peak/lowest pre-computed. Leading with the answer first so the
        # embedding picks up "which month had the highest" type queries.
        for year, g in df.groupby('year'):
            monthly = g.groupby('month')['units_sold'].sum()
            peak_m = int(monthly.idxmax())
            peak_v = int(monthly.max())
            low_m = int(monthly.idxmin())
            low_v = int(monthly.min())
            month_lines = ", ".join(
                f"{month_names[int(m)]}: {int(u):,} units"
                for m, u in monthly.items()
            )
            docs.append({
                "content": (
                    f"The month with the highest sales in {year} was "
                    f"{month_names[peak_m]} with {peak_v:,} units. "
                    f"The month with the lowest sales in {year} was "
                    f"{month_names[low_m]} with {low_v:,} units. "
                    f"Full monthly breakdown for {year}: {month_lines}."
                ),
                "metadata": {"scope": "year_monthly_breakdown", "year": int(year)},
            })

        # Per-year regional breakdown — leading with the top region.
        for year, g in df.groupby('year'):
            regional = g.groupby('region')['units_sold'].sum().sort_values(ascending=False)
            top_r = regional.index[0]
            top_v = int(regional.iloc[0])
            reg_lines = ", ".join(
                f"{r}: {int(u):,} units" for r, u in regional.items()
            )
            docs.append({
                "content": (
                    f"The top-performing region in {year} was {top_r} "
                    f"with {top_v:,} units. The region with the most sales "
                    f"in {year} was {top_r}. Full regional breakdown for "
                    f"{year}: {reg_lines}."
                ),
                "metadata": {"scope": "year_regional_breakdown", "year": int(year)},
            })

        # Per-year category breakdown — leading with the top category.
        for year, g in df.groupby('year'):
            categorical = g.groupby('category')['units_sold'].sum().sort_values(ascending=False)
            top_c = categorical.index[0]
            top_v = int(categorical.iloc[0])
            cat_lines = ", ".join(
                f"{c}: {int(u):,} units" for c, u in categorical.items()
            )
            docs.append({
                "content": (
                    f"The top-performing category in {year} was {top_c} "
                    f"with {top_v:,} units. The category with the most sales "
                    f"in {year} was {top_c}. Full category breakdown for "
                    f"{year}: {cat_lines}."
                ),
                "metadata": {"scope": "year_category_breakdown", "year": int(year)},
            })

        # Per-brand totals (across all years/regions)
        if 'brand' in df.columns:
            for brand, g in df.groupby('brand'):
                units = g['units_sold'].sum()
                ap = g['price_unit'].mean()
                cats = ", ".join(sorted(g['category'].unique().tolist()))
                docs.append({
                    "content": (
                        f"Brand {brand} had total sales of {units:,.0f} units at average "
                        f"price ${ap:.2f} per unit. Categories: {cats}."
                    ),
                    "metadata": {"scope": "brand", "brand": brand},
                })

        # === Granular facts (original docs, preserved) ===

        # Daily summary
        for _, row in df.groupby("date").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "stock_available": "mean",
            "promotion_flag": "sum"
        }).reset_index().iterrows():
            docs.append({
                "content": (
                    f"On {row['date'].strftime('%Y-%m-%d')}, sales were "
                    f"{row['units_sold']:.0f} units, avg price ${row['price_unit']:.2f}, "
                    f"stock {row['stock_available']:.0f}, {row['promotion_flag']} promos."
                ),
                "metadata": {"date": row['date'].strftime('%Y-%m-%d')},
            })

        # Per-SKU
        for _, row in df.groupby("sku").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "region": "nunique"
        }).reset_index().iterrows():
            docs.append({
                "content": (
                    f"Product {row['sku']} sold {row['units_sold']:.0f} units across "
                    f"{row['region']} regions at avg price ${row['price_unit']:.2f}."
                ),
                "metadata": {"sku": row['sku']},
            })

        # Per-region
        for _, row in df.groupby("region").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "sku": "nunique"
        }).reset_index().iterrows():
            docs.append({
                "content": (
                    f"Region {row['region']} sold {row['units_sold']:.0f} units, "
                    f"{row['sku']} products, avg price ${row['price_unit']:.2f}."
                ),
                "metadata": {"region": row['region']},
            })

        return docs

    # =============== Vector Store ===============
    def create_vector_store(self, docs):
        texts = [d["content"] for d in docs]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))
        self.vector_store, self.documents, self.embeddings = index, docs, embeddings

    def load_vector_store(self):
        """Load existing vector store, documents, and embeddings if available."""
        try:
            faiss_index_path = self.vector_store_dir / "faiss_index.bin"
            if not faiss_index_path.exists():
                logger.warning("No FAISS index found.")
                return False
            self.vector_store = faiss.read_index(str(faiss_index_path))
            # Load documents
            with open(self.vector_store_dir / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            # Load embeddings
            with open(self.vector_store_dir / "embeddings.pkl", "rb") as f:
                self.embeddings = pickle.load(f)
            logger.info("Vector store loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    # =============== QA ===============
    # =============== QA ===============
    def answer_question(self, question, context=None):
        """Legacy method returning just the answer string."""
        result = self.answer_query(question, context)
        return result["answer"]

    def answer_query(self, question, context=None):
        """
        Answer a query and return both answer and sources.

        Routing: every question goes through the LangChain LCEL chain
        (ChatPromptTemplate | ChatGroq | StrOutputParser) over FAISS-retrieved
        context. The previous keyword-based analytical fast-path was removed
        because it intercepted filtered questions and returned generic
        unfiltered aggregations. Trusting the LLM with grounded context plus
        a low-temperature prompt produces more accurate, verbose answers.

        Returns: dict {'answer': str, 'sources': list}
        """
        sources = []

        # RAG: FAISS retrieval + LangChain Groq generation
        if self.vector_store is not None and self.documents is not None:
            top_k = self.rag_config.get("top_k", 5)
            query_embedding = self.embedding_model.encode([question]).astype("float32")
            D, I = self.vector_store.search(query_embedding, k=top_k)
            retrieved_docs = [self.documents[i] for i in I[0]]
            context = "\n".join([doc["content"] for doc in retrieved_docs])
            sources = [doc["content"][:200] + "..." for doc in retrieved_docs[:3]]

            try:
                answer = self.qa_chain.invoke({"context": context, "question": question})
                return {"answer": answer.strip(), "sources": sources}
            except Exception as e:
                err_str = str(e).lower()
                logger.error(f"LangChain Groq generation failed: {e}")
                if "connection" in err_str or "timeout" in err_str:
                    hint = (
                        "Transient connection issue talking to Groq — usually clears up "
                        "in a few seconds. Please retry the question."
                    )
                elif "401" in err_str or "invalid" in err_str and "key" in err_str:
                    hint = (
                        "Groq rejected the API key. Confirm GROQ_API_KEY in your .env "
                        "file is current and not revoked."
                    )
                elif "429" in err_str or "rate" in err_str:
                    hint = (
                        "Groq rate limit hit. Wait ~30 seconds and retry."
                    )
                else:
                    hint = f"Generation failed: {e}"
                return {"answer": hint, "sources": sources}

        return {"answer": "I couldn't find relevant information to answer that question.", "sources": []}
    
    def run_rag_pipeline(self):
        """
        Run the complete RAG pipeline setup and testing.
        Returns: dict with pipeline results
        """
        logger.info("Starting RAG pipeline execution...")
        
        try:
            # Step 1: Load sales data
            logger.info("Loading sales data...")
            df = self.load_sales_data()
            logger.info(f"Loaded {len(df)} records")
            
            # Step 2: Create documents
            logger.info("Creating documents from sales data...")
            docs = self.create_documents(df)
            logger.info(f"Created {len(docs)} documents")
            
            # Step 3: Create vector store
            logger.info("Building vector store...")
            self.create_vector_store(docs)
            
            # Step 4: Save vector store
            logger.info("Saving vector store...")
            faiss.write_index(self.vector_store, str(self.vector_store_dir / "faiss_index.bin"))
            
            with open(self.vector_store_dir / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(self.vector_store_dir / "embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings, f)
            
            logger.info("Vector store saved successfully")
            
            # Step 5: Test with sample queries
            logger.info("Testing RAG system with sample queries...")
            test_queries = [
                "What were the total sales in 2023?",
                "Which product had the highest sales?",
                "What is the average price across all regions?",
                "How do sales compare with and without promotions?",
                "What are the seasonal sales patterns?"
            ]
            
            test_results = []
            for query in test_queries:
                result = self.answer_query(query)
                test_results.append({
                    "query": query,
                    "answer": result
                })
            
            # Save test results
            with open(self.reports_dir / "rag_test_results.json", "w") as f:
                json.dump(test_results, f, indent=2)
            
            logger.info("RAG testing completed")
            
            # Step 6: Generate summary report
            report = self.generate_report(df, "summary")
            self.save_report(report, "summary")
            
            return {
                'status': 'success',
                'documents_created': len(docs),
                'vector_store_size': len(self.documents),
                'test_queries_run': len(test_results),
                'data_records': len(df)
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    # =============== Reporting ===============
    def generate_report(self, df, report_type="summary"):
        if report_type == "summary":
            return self.generate_summary_report(df)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def generate_summary_report(self, df):
        # Simple text-based report
        total_sales = df["units_sold"].sum()
        total_revenue = (df["units_sold"] * df["price_unit"]).sum()
        num_transactions = df.shape[0]
        report = f"FMCG Sales Report Summary\n"
        report += f"Total Sales: {total_sales:.0f} units\n"
        report += f"Total Revenue: ${total_revenue:,.2f}\n"
        report += f"Number of Transactions: {num_transactions}\n"
        logger.info("Generated summary report.")
        return report

    def save_report(self, report, report_type="summary"):
        path = self.reports_dir / f"{report_type}_report.txt"
        with open(path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {path}.")