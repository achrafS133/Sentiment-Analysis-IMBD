
install:
	pip install -r requirements.txt

test:
	pytest tests/

format:
	black src/

lint:
	flake8 src/

train:
	python main.py

api-dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
