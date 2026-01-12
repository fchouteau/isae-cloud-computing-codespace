app:
	uvicorn app:app --reload
build:
	docker build -f Dockerfile -t europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-backend:1.0 .
run:
	docker run --rm -p 8000:8000 europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-backend:1.0
push:
	docker push europe-docker.pkg.dev/isae-sdd-481407/isae-sdd-de-2526-docker/yolo-v11-backend:1.0
