FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py model.py state_dictmodel.pth mean.pt std.pt Y_mean.pt Y_std.pt Statusonehot_encoder.pkl Furnishedonehot_encoder.pkl Addressonehot_encoder.pkl Property_Typeonehot_encoder.pkl .
EXPOSE 7860  
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:7860"]  # Change to 7860