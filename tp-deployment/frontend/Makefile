app:
	streamlit run app.py
build:
	docker build -f Dockerfile -t eu.gcr.io/third-ridge-138414/yolo-v5-streamlit:1.5 .
run:
	docker run --rm -p 8501:8501 --network="host" eu.gcr.io/third-ridge-138414/yolo-v5-streamlit:1.5
push:
	docker push eu.gcr.io/third-ridge-138414/yolo-v5-streamlit:1.5
