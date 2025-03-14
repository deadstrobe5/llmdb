# Deploying the NL-to-SQL Web App

This document provides simple instructions for deploying the Natural Language to SQL web app so it can be accessed by others online.

## Option 1: Streamlit Community Cloud (Easiest)

1. **Push your code to GitHub**:
   - Create a GitHub repository
   - Push your code to the repository
   - Make sure your `.env` file is not included (add it to `.gitignore`)

2. **Sign up for Streamlit Community Cloud**:
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

3. **Deploy the app**:
   - Click "New app"
   - Select your repository, branch, and `app.py` as the main file
   - Add your secrets (OPENAI_API_KEY) in the "Advanced settings" section
   - Click "Deploy"

4. **Share the link**:
   - Once deployed, you'll get a public URL like: `https://yourusername-app-name.streamlit.app`
   - Share this link with your friend

## Option 2: Deploy with Render

1. **Sign up for Render**:
   - Go to [https://render.com/](https://render.com/)
   - Create a free account

2. **Deploy a new Web Service**:
   - Connect your GitHub repository
   - Select "Python" as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Add your OPENAI_API_KEY as an environment variable
   - Deploy the service

3. **Share the app URL**:
   - Render will provide a URL for your app (like `https://your-app-name.onrender.com`)
   - Share this URL with your friend

## Before Deploying

1. Make sure your vector database is initialized: run `python init_vector_db.py`
2. Test your app locally: run `streamlit run app.py`
3. Make sure all required packages are in `requirements.txt`
4. Add your OPENAI_API_KEY as a secret/environment variable on the hosting platform

## Local Testing

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the vector database (if not already done)
python init_vector_db.py

# Run the app
streamlit run app.py
```

Your app will be accessible at `http://localhost:8501` 