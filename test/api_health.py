#!/usr/bin/env python3
import os
import logging
from dotenv import load_dotenv
import requests
from supabase import create_client
import redis
from celery import Celery
from kombu import Connection
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def setup_logging():
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def test_env_var(name, value):
    if not value:
        logging.warning(f"{name} not set")
        return False
    return True

def test_openai(api_key, label):
    if not test_env_var(label, api_key):
        return
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            logging.info(f"{label} key live")
        else:
            logging.warning(f"{label} key failed: HTTP {r.status_code}")
    except Exception as e:
        logging.warning(f"{label} key failed: {e}")

def test_gemini(api_key):
    if not test_env_var("GEMINI_API_KEY", api_key):
        return
    url = "https://generativelanguage.googleapis.com/v1beta2/models"
    params = {"key": api_key}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            logging.info("Gemini API key live")
        else:
            logging.warning(f"Gemini API key failed: HTTP {r.status_code}")
    except Exception as e:
        logging.warning(f"Gemini API key failed: {e}")

def test_deepseek(api_key):
    if not test_env_var("DEEPSEEK_API", api_key):
        return
    # Replace with your real DeepSeek status endpoint if you have one
    url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.ai/v1/status")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.ok:
            logging.info("DeepSeek API key live")
        else:
            logging.warning(f"DeepSeek API key failed: HTTP {r.status_code}")
    except Exception as e:
        logging.warning(f"DeepSeek API key failed: {e}")

def test_supabase(url, key):
    if not (test_env_var("SUPABASE_URL", url) and test_env_var("SUPABASE_SERVICE_ROLE_KEY", key)):
        return
    try:
        client = create_client(url, key)
        # try a minimal query
        resp = client.from_("projects").select("id", count="exact", limit=1).execute()
        if resp.error:
            logging.warning(f"Supabase query failed: {resp.error.message}")
        else:
            logging.info("Supabase connection live")
    except Exception as e:
        logging.warning(f"Supabase connection failed: {e}")

def test_redis(url):
    if not test_env_var("REDIS_LABS_URL_AND_PASS", url):
        return
    try:
        r = redis.Redis.from_url(url)
        r.ping()
        logging.info("Redis Labs connection live")
    except Exception as e:
        logging.warning(f"Redis Labs connection failed: {e}")

def test_amqp(broker_url):
    if not test_env_var("CLOUDAMQP_PUBLIC_ENDPOINT", broker_url):
        return
    try:
        with Connection(broker_url) as conn:
            conn.connect()
        logging.info("CloudAMQP connection live")
    except Exception as e:
        logging.warning(f"CloudAMQP connection failed: {e}")

def test_celery(broker_url, backend_url):
    if not (test_env_var("CLOUDAMQP_PUBLIC_ENDPOINT", broker_url) and test_env_var("REDIS_LABS_URL_AND_PASS", backend_url)):
        return
    try:
        app = Celery('ping', broker=broker_url, backend=backend_url)
        # this will raise if broker/backend unreachable
        reply = app.control.ping(timeout=5)
        logging.info("Celery broker/backend reachable" + (", workers: " + str(reply) if reply else ", no workers responded"))
    except Exception as e:
        logging.warning(f"Celery broker/backend connection failed: {e}")

def test_aws_iam(access_key, secret_key):
    if not (test_env_var("AWS_IAM_ACCESS_KEY", access_key) and test_env_var("AWS_IAM_SECRET", secret_key)):
        return
    try:
        sts = boto3.client(
            'sts',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        ident = sts.get_caller_identity()
        logging.info(f"AWS IAM credentials valid (Account: {ident['Account']})")
    except Exception as e:
        logging.warning(f"AWS IAM credentials failed: {e}")

def test_s3(access_key, secret_key, bucket):
    if not test_env_var("AWS_S3_BUCKET_NAME", bucket):
        return
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3.head_bucket(Bucket=bucket)
        logging.info(f"AWS S3 bucket '{bucket}' accessible")
    except Exception as e:
        logging.warning(f"AWS S3 bucket access failed: {e}")

def test_cloudfront(domain):
    if not test_env_var("AWS_CLOUDFRONT_DOMAIN_PROD", domain):
        return
    url = f"https://{domain}"
    try:
        r = requests.head(url, timeout=5)
        if r.status_code < 400:
            logging.info(f"AWS CloudFront domain reachable: HTTP {r.status_code}")
        else:
            logging.warning(f"AWS CloudFront domain returned HTTP {r.status_code}")
    except Exception as e:
        logging.warning(f"AWS CloudFront domain unreachable: {e}")

def main():
    setup_logging()
    load_dotenv()

    # load env vars (note typo fallback for SUPABASE_PROJECT_URL)
    openai_prod = os.getenv("OPENAI_API_PROD_KEY")
    openai_test = os.getenv("OPENAI_API_TEST_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API")
    supabase_url = os.getenv("SUPABASE_PROJECT_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    redis_url = os.getenv("REDIS_LABS_URL_AND_PASS")
    amqp_url = os.getenv("CLOUDAMQP_PUBLIC_ENDPOINT")
    aws_key = os.getenv("AWS_IAM_ACCESS_KEY")
    aws_secret = os.getenv("AWS_IAM_SECRET")
    s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")
    cf_domain = os.getenv("AWS_CLOUDFRONT_DOMAIN_PROD")

    # run tests
    test_openai(openai_prod, "OpenAI Prod")
    test_openai(openai_test, "OpenAI Test")
    test_gemini(gemini_key)
    test_deepseek(deepseek_key)
    test_supabase(supabase_url, supabase_key)
    test_redis(redis_url)
    test_amqp(amqp_url)
    test_celery(amqp_url, redis_url)
    test_aws_iam(aws_key, aws_secret)
    test_s3(aws_key, aws_secret, s3_bucket)
    test_cloudfront(cf_domain)

if __name__ == "__main__":
    main()
