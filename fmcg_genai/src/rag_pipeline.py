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
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an FMCG sales analyst. Answer the user's question using ONLY the "
             "context below. If the context does not contain enough information, say so "
             "honestly. Be concise (2-3 sentences) and quantitative when possible.\n\n"
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
        docs = []
        # Daily summary
        for _, row in df.groupby("date").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "stock_available": "mean",
            "promotion_flag": "sum"
        }).reset_index().iterrows():
            docs.append({"content": f"On {row['date'].strftime('%Y-%m-%d')}, sales were {row['units_sold']:.0f} units, avg price ${row['price_unit']:.2f}, stock {row['stock_available']:.0f}, {row['promotion_flag']} promos.",
                         "metadata": {"date": row['date'].strftime('%Y-%m-%d')}})
        # Product summary
        for _, row in df.groupby("sku").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "region": "nunique"
        }).reset_index().iterrows():
            docs.append({"content": f"Product {row['sku']} sold {row['units_sold']:.0f} units across {row['region']} regions at avg price ${row['price_unit']:.2f}.",
                         "metadata": {"sku": row['sku']}})
        # Region summary
        for _, row in df.groupby("region").agg({
            "units_sold": "sum",
            "price_unit": "mean",
            "sku": "nunique"
        }).reset_index().iterrows():
            docs.append({"content": f"Region {row['region']} sold {row['units_sold']:.0f} units, {row['sku']} products, avg price ${row['price_unit']:.2f}.",
                         "metadata": {"region": row['region']}})
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
        Returns: dict {'answer': str, 'sources': list}
        """
        sources = []
        
        # First, try to answer analytically from the actual data
        try:
            df = self.load_sales_data()
            analytical_answer = self._generate_analytical_answer(question, df)
            if analytical_answer:
                return {
                    "answer": analytical_answer,
                    "sources": ["Computed from sales data analytics"]
                }
        except Exception as e:
            logger.warning(f"Analytical answer generation failed: {e}")
        
        # Fallback to RAG: FAISS retrieval + LangChain Groq generation
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
                logger.error(f"LangChain Groq generation failed: {e}")
                return {
                    "answer": (
                        "I retrieved relevant context but couldn't generate an answer. "
                        "Check that GROQ_API_KEY is set in your .env file."
                    ),
                    "sources": sources,
                }

        return {"answer": "I couldn't find relevant information to answer that question.", "sources": []}
    
    def _generate_analytical_answer(self, question, df):
        """Generate analytical answers based on actual data analysis"""
        question_lower = question.lower()
        
        # Total sales questions
        if any(word in question_lower for word in ['total sales', 'overall sales', 'sales volume']):
            if '2023' in question_lower:
                sales_2023 = df[df['date'].dt.year == 2023]['units_sold'].sum()
                return f"Total sales in 2023 were {sales_2023:,} units."
            elif '2024' in question_lower:
                sales_2024 = df[df['date'].dt.year == 2024]['units_sold'].sum()
                return f"Total sales in 2024 were {sales_2024:,} units."
            else:
                total_sales = df['units_sold'].sum()
                return f"Total sales across all periods were {total_sales:,} units."
        
        # Top product questions
        if any(word in question_lower for word in ['highest sales', 'top product', 'best selling', 'most sold']):
            top_product = df.groupby('sku')['units_sold'].sum().idxmax()
            top_sales = df.groupby('sku')['units_sold'].sum().max()
            return f"The product with the highest sales is {top_product} with {top_sales:,} units sold."
        
        # Average price questions
        if 'average price' in question_lower or 'avg price' in question_lower:
            avg_price = df['price_unit'].mean()
            return f"The average price per unit is ${avg_price:.2f}."
        
        # Regional performance
        if 'region' in question_lower and any(word in question_lower for word in ['best', 'top', 'highest']):
            top_region = df.groupby('region')['units_sold'].sum().idxmax()
            region_sales = df.groupby('region')['units_sold'].sum().max()
            return f"The best performing region is {top_region} with {region_sales:,} units sold."
        
        # Price vs sales correlation
        if 'price' in question_lower and any(word in question_lower for word in ['affect', 'impact', 'correlation', 'relationship']):
            correlation = df[['price_unit', 'units_sold']].corr().iloc[0, 1]
            avg_price = df['price_unit'].mean()
            high_price_sales = df[df['price_unit'] > avg_price]['units_sold'].mean()
            low_price_sales = df[df['price_unit'] <= avg_price]['units_sold'].mean()
            
            if correlation < -0.3:
                trend = "negative correlation"
                explanation = "higher prices tend to result in lower sales volume"
            elif correlation > 0.3:
                trend = "positive correlation"
                explanation = "higher prices are associated with higher sales volume"
            else:
                trend = "weak correlation"
                explanation = "price has minimal direct impact on sales volume"
            
            return f"There is a {trend} ({correlation:.2f}) between price and sales. Products priced above average (${avg_price:.2f}) sell {high_price_sales:.0f} units/day on average, while those below average sell {low_price_sales:.0f} units/day. This suggests {explanation}."
        
        # Seasonal patterns
        if 'seasonal' in question_lower or 'pattern' in question_lower:
            monthly_avg = df.groupby(df['date'].dt.month)['units_sold'].mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            return f"Seasonal analysis shows peak sales in {month_names[peak_month]} ({monthly_avg[peak_month]:.0f} units/day) and lowest in {month_names[low_month]} ({monthly_avg[low_month]:.0f} units/day)."
        
        # Peak month
        if 'peak' in question_lower and 'month' in question_lower:
            if '2024' in question_lower:
                df_2024 = df[df['date'].dt.year == 2024]
                monthly_sales = df_2024.groupby(df_2024['date'].dt.month)['units_sold'].sum()
            else:
                monthly_sales = df.groupby(df['date'].dt.month)['units_sold'].sum()
            
            peak_month = monthly_sales.idxmax()
            peak_sales = monthly_sales.max()
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            year_str = "in 2024" if '2024' in question_lower else "overall"
            return f"The peak sales month {year_str} was {month_names[peak_month]} with {peak_sales:,} units sold."
        
        # Stock availability impact
        if 'stock' in question_lower and any(word in question_lower for word in ['impact', 'affect', 'availability']):
            high_stock = df[df['stock_available'] > df['stock_available'].median()]['units_sold'].mean()
            low_stock = df[df['stock_available'] <= df['stock_available'].median()]['units_sold'].mean()
            diff_pct = ((high_stock - low_stock) / low_stock * 100)
            return f"Stock availability has a significant impact on sales. Products with above-median stock levels sell {high_stock:.0f} units/day on average, compared to {low_stock:.0f} units/day for below-median stock - a {diff_pct:.1f}% difference."
        
        return None  # No analytical answer found, will fallback to RAG

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