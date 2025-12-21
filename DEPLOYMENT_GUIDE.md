# 🚀 Quick Deployment Guide for Hugging Face Spaces

## Why Hugging Face Spaces?
- **16GB RAM** (vs Render's 750MB limit)
- **Free forever** - No credit card required
- **Perfect for ML/AI** projects
- **Auto-deployment** from GitHub

## Step-by-Step Deployment

### 1. Create a Hugging Face Account
- Go to [huggingface.co](https://huggingface.co/)
- Sign up for a free account

### 2. Create a New Space
- Click on your profile → **"New Space"**
- **Name**: `fmcg-demand-forecasting` (or your choice)
- **License**: MIT
- **SDK**: Select **"Streamlit"**
- **Hardware**: **CPU basic** (free tier)

### 3. Connect to GitHub
- Choose **"Import from GitHub"**
- Enter your repo URL: `https://github.com/Ameya-Bhingurde/FMCG-Demand-Forecasting-with-RAG-`
- Click **"Import"**

### 4. Configure Space Settings
The README.md already contains the required YAML frontmatter:
```yaml
---
title: FMCG Demand Forecasting with RAG
sdk: streamlit
sdk_version: "1.25.0"
app_file: fmcg_genai/src/dashboard_app_enhanced.py
---
```

### 5. Wait for Build
- Hugging Face will automatically:
  - Install dependencies from `fmcg_genai/requirements.txt`
  - Run the pipeline to generate data files
  - Train models
  - Create vector store
  - Launch the dashboard

### 6. Access Your App
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/fmcg-demand-forecasting`

## Important Notes

### Large Files
⚠️ The processed data files (`features.csv`, `test_features.csv`) are **excluded from Git** because they exceed GitHub's 100MB limit.

**Solution**: The `run_pipeline.py` script will automatically generate these files during deployment on Hugging Face Spaces.

### First Launch
- **First deployment may take 10-15 minutes** as it processes data and trains models
- Subsequent launches will be faster (models are cached)

### Memory Requirements
- **Minimum**: 2GB RAM
- **Recommended**: 4GB+ RAM
- **Hugging Face Free Tier**: 16GB RAM ✅

## Alternative: Manual Deployment

If you prefer manual setup:

### Option 1: Clone Directly to Hugging Face
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME
git remote add github https://github.com/Ameya-Bhingurde/FMCG-Demand-Forecasting-with-RAG-.git
git pull github main
git push origin main
```

### Option 2: Use Hugging Face CLI
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create fmcg-demand-forecasting --type space --space_sdk streamlit
git push https://huggingface.co/spaces/YOUR_USERNAME/fmcg-demand-forecasting main
```

## Troubleshooting

### Build Fails
- Check the build logs in Hugging Face Spaces
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility (3.8+)

### App Won't Start
- Check if `run_pipeline.py` completed successfully
- Verify models were created in `models/` directory
- Check vector store exists in `vector_store/` directory

### Out of Memory
- Reduce model size in `config.yaml`
- Use lighter embedding model
- Upgrade to Hugging Face Pro (48GB RAM)

## Support
- **GitHub Issues**: [Report bugs](https://github.com/Ameya-Bhingurde/FMCG-Demand-Forecasting-with-RAG-/issues)
- **Hugging Face Discussions**: Available after deployment

---

**Ready to deploy?** Follow the steps above and your app will be live in minutes! 🚀
