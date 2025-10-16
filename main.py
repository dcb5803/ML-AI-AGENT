import pandas as pd, numpy as np, mlflow, uvicorn, os, json
from fastapi import FastAPI, Request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from langchain.agents import initialize_agent, Tool
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Synthetic Dataset ===
X = np.random.rand(100, 3)
y = X @ np.array([3.5, -2.1, 1.7]) + np.random.randn(100) * 0.1
df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3"])
df["target"] = y

# === MLflow Setup ===
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("ml_pipeline_agent")

# === Model Training ===
def train_model():
    with mlflow.start_run(run_name="train"):
        model = LinearRegression().fit(df[["feat1", "feat2", "feat3"]], df["target"])
        preds = model.predict(df[["feat1", "feat2", "feat3"]])
        mse = mean_squared_error(df["target"], preds)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        with open("latest_log.json", "w") as f:
            json.dump({"mse": mse}, f)
        return mse

# === LLM Agent Setup ===
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

def check_logs(_):
    with open("latest_log.json") as f:
        log = json.load(f)
    if log["mse"] > 0.2:
        return f"Performance dropped (MSE={log['mse']}). Retraining triggered.\n" + str(train_model())
    return f"Model performance is stable (MSE={log['mse']}). No action needed."

tools = [Tool(name="CheckLogs", func=check_logs, description="Check logs and retrain if needed")]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# === FastAPI Deployment ===
app = FastAPI()

@app.get("/")
def root():
    return {"message": "ML Pipeline Agent is running."}

@app.post("/trigger")
async def trigger(request: Request):
    body = await request.json()
    query = body.get("query", "Check logs and retrain if needed")
    response = agent.run(query)
    return {"response": response}

# === Entry Point ===
if __name__ == "__main__":
    train_model()  # Initial training
    uvicorn.run(app, host="0.0.0.0", port=8000)
