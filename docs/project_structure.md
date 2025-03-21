# Project Documentation 

## Structure Overview

```

root
|-- src/
    |-- service.py

|-- tasks/
    |-- __init__.py
    |-- models.py
|-- utils/
    |-- __init__.py
    |-- models.py
|-- docs/
    |-- documentation.md
|-- main.py
|-- .env

```

## Directories 

### `utils/`

Utilities folder with helper functions for 

### `tasks/`

Tasks dir contains task functions that are processed to be send and executed my the Celery App

celery_app.py: this script inits a celery object which is use throughough the rest of the project

For testing the celery service
```bash
celery task.celery_app.py
```

In production (i.e. Render.io):

```bash
startCommand: "celery -A tasks.celery_app worker --loglevel=info --concurrency=2"
```

### `.env`

contains