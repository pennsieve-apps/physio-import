.PHONY: help test fixture run run-demos lambda-check clean

SERVICE_NAME ?= physio-import
PYTHON ?= python3
IMAGE ?= physio-import:dev

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test         - pytest: synthetic recordings in many formats through the processor (no Docker)"
	@echo "make fixture      - write the synthetic fixture to data/fixture/ (recordings/ and truth.json)"
	@echo "make run          - build the image and run it in Docker over data/fixture/recordings/, writing to data/output/"
	@echo "make run-demos    - run the image over the demo data of workflow-physiology (validation/run-demos.sh)"
	@echo "make lambda-check - run the image as Lambda runs it: read-only, any uid, HOME/TMPDIR under /tmp"
	@echo "make clean        - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-demos:
	docker build --platform linux/amd64 -t $(IMAGE) .
	DOCKER=1 IMAGE=$(IMAGE) validation/run-demos.sh

lambda-check: fixture
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-lambda && mkdir -p data/output-lambda && chmod 777 data/output-lambda
	docker run --rm --platform linux/amd64 --memory 2g --read-only --user 65534:65534 --tmpfs /tmp:mode=1777 \
		-e HOME=/tmp/w/home -e TMPDIR=/tmp/w/tmp -e INPUT_DIR=/tmp/s/input -e OUTPUT_DIR=/tmp/s/output \
		-v "$(CURDIR)/data/fixture/recordings":/tmp/s/input:ro -v "$(CURDIR)/data/output-lambda":/tmp/s/output $(IMAGE)

clean:
	rm -rf data/output data/output-lambda data/fixture
