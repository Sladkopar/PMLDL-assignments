# PMLDL-assignments

# Configuration (for Windows)

In `docker-compose.yml`, there is a mounted directory `${PWD}/../../models:/app/models`, which doesn't work without use of absolute path.

```sh
cd code/deployment
$env:PWD = (Get-Location).Path
docker-compose up --build
```

The code above runs the docker container with two images: `api`(FastAPI) and `app`(StreamLit). 
API works on http://0.0.0.0:80 and StreamLit is on http://0.0.0.0:8501.

# Usage

1) On [StreamLit](http://0.0.0.0:8501), click `Browse Files`
2) Choose some photo of a person (picture of a face)
3) After classification, model there will be a result like `Prediction: The person is not wearing glasses.` below.