# LegalCaseAIApp
pdf rag and generative podcast 

Conda envirionment (legalnote python=3.12)



## Rough Project Structure

```
root
|-- app/
    |-- __init__.py
    |-- main.py             # fastAPI app
|-- tasks/
    |-- __init__.py
    |-- celery_app.py
    |-- chat_tasks.py
    |-- upload_tasks.py
    |-- ...
|    
|-- utils/
    |-- __init__.py
    |-- supabase_utils.py
|-- docs/
    |-- documentation.md
|-- .env
```