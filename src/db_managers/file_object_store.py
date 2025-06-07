import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
# import streamlit as st # UI feedback should be handled by calling code in app layer
from werkzeug.utils import secure_filename # For sanitizing filenames
import uuid # For generating unique filenames
# from dotenv import load_dotenv # No longer needed here, app_config handles it

# Import R2 configuration constants from app_config
from src.config.app_config import (
    R2_ENDPOINT_URL,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    R2_UPLOAD_FOLDER # Using this for consistency if uploads should go to a subfolder
)

# load_dotenv() # Handled by app_config.py

_r2_client_instance = None # Cached client instance

def get_r2_client():
    """
    Initializes and returns a boto3 S3 client configured for Cloudflare R2.
    Caches the client instance for efficiency.
    Returns None if configuration is missing or an error occurs.
    """
    global _r2_client_instance
    if _r2_client_instance:
        return _r2_client_instance

    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]): # Check all required R2 vars
        print("INFO: Cloudflare R2 environment variables not fully set (via app_config). R2 client will not be initialized.")
        return None

    try:
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=R2_ENDPOINT_URL, # Use imported constant
            aws_access_key_id=R2_ACCESS_KEY_ID, # Use imported constant
            aws_secret_access_key=R2_SECRET_ACCESS_KEY, # Use imported constant
            region_name='auto',
        )
        s3_client.list_buckets() # Quick check for credential validity
        _r2_client_instance = s3_client
        print("INFO: Cloudflare R2 client initialized successfully.")
        return _r2_client_instance
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"ERROR: R2 credentials error: {e}")
        return None
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "InvalidAccessKeyId" or error_code == "SignatureDoesNotMatch":
            print(f"ERROR: R2 client authentication failed: {error_code}. Check R2 credentials.")
        else:
            print(f"ERROR: R2 ClientError during client initialization: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error creating R2 client: {e}")
        return None

def upload_file_to_r2(file_obj, bucket_name: str | None = None, object_name: str | None = None) -> str | None:
    """
    Uploads a file object to Cloudflare R2.
    Args:
        file_obj: The file-like object to upload (e.g., Streamlit UploadedFile).
        bucket_name: The R2 bucket name. Defaults to R2_BUCKET_NAME from .env.
        object_name: The desired object name (path) in R2. If None, a unique name
                     based on the original filename is generated.
    Returns:
        The object name (key) in R2 if successful, otherwise None.
    """
    client = get_r2_client()
    if not client:
        print("ERROR: Cloudflare R2 client not configured in file_object_store. Cannot upload file.")
        return None

    actual_bucket_name = bucket_name if bucket_name else R2_BUCKET_NAME
    if not actual_bucket_name: # This refers to the R2_BUCKET_NAME constant
        print("ERROR: R2_BUCKET_NAME is not set (via app_config). Cannot upload file.")
        return None

    if object_name is None:
        original_filename = "untitled"
        if hasattr(file_obj, 'name') and file_obj.name:
            original_filename = file_obj.name

        sanitized_name = secure_filename(original_filename)
        # Use R2_UPLOAD_FOLDER from config
        final_object_name = f"{R2_UPLOAD_FOLDER.strip('/')}/{uuid.uuid4()}_{sanitized_name}"
    else:
        final_object_name = object_name

    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0) # Ensure reading from the beginning

        client.upload_fileobj(file_obj, actual_bucket_name, final_object_name) # actual_bucket_name uses R2_BUCKET_NAME constant

        print(f"INFO: File {final_object_name} uploaded to R2 bucket {actual_bucket_name}.")
        return final_object_name
    except ClientError as e:
        print(f"ERROR: ClientError uploading file '{final_object_name}' to R2 bucket '{actual_bucket_name}': {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during R2 upload for '{final_object_name}' to bucket '{actual_bucket_name}': {e}")
        return None

if __name__ == '__main__':
    print("Testing file_object_store.py...")

    # Test get_r2_client (which now uses constants from app_config)
    client = get_r2_client()

    # R2_BUCKET_NAME is imported from app_config
    if client and R2_BUCKET_NAME:
        print(f"R2 Client appears to be initialized. Target bucket: {R2_BUCKET_NAME}")
        from io import BytesIO

        class MockUploadedFile: # Simplified mock
            def __init__(self, name, content_bytes):
                self.name = name
                self.file_obj = BytesIO(content_bytes)
            def read(self, size=-1): return self.file_obj.read(size)
            def seek(self, offset, whence=0): return self.file_obj.seek(offset, whence)

        dummy_file = MockUploadedFile("test_r2_upload.txt", b"Hello R2 from file_object_store!")
        print(f"Attempting to upload '{dummy_file.name}' (using R2_UPLOAD_FOLDER: '{R2_UPLOAD_FOLDER}')...")

        object_key = upload_file_to_r2(dummy_file) # bucket_name will default to R2_BUCKET_NAME
        if object_key:
            print(f"SUCCESS: Uploaded to R2. Object Key: {object_key}")
        else:
            print("FAILURE: Upload to R2 failed. Check server logs and .env configuration (via app_config).")
    elif not R2_BUCKET_NAME:
        print("R2_BUCKET_NAME (via app_config) is not set. Cannot run upload test.")
    else:
        print("R2 client initialization failed. Cannot run upload test. Check .env variables (via app_config) and R2 connection.")

    print("file_object_store.py test finished.")
