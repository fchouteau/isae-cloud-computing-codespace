app:
	streamlit run app.py
build:
	docker build -f Dockerfile -t europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-frontend:1.0 .
run:
	docker run --rm -p 8501:8501 --network="host" europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-frontend:1.0
push:
	docker push europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-frontend:1.0
