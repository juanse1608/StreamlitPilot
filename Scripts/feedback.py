import streamlit as st
from google.cloud import storage, bigquery 
from google.cloud.bigquery.schema import SchemaField
from google.oauth2 import service_account
from PIL import Image
import json
import io
import os
import pandas as pd

SEND_FEEDBACK = True

class GCP_USER:

    def __init__(self, credentials):
        self.credentials = service_account.Credentials.from_service_account_info(credentials)
        self.storage_cl = storage.Client(credentials=self.credentials)
        self.bigq_cl = bigquery.Client(credentials=self.credentials)

    def load_image_and_caption(self, feedback_query, caption, uploaded_image, storage_bucket_name, bigquery_table_name):
        
        # Image
        with open(feedback_query, "r") as query_file:
            query = query_file.read()
        df_max_id = self.bigq_cl.query(query).to_dataframe()
        df_max_id['MAX_ID'].fillna(0, inplace=True)
        new_id = int(df_max_id.loc[0,'MAX_ID']+1)
        img_file = f'{str(new_id)}.jpg'
        img = Image.open(io.BytesIO(uploaded_image))
        img.save(img_file)
        blob = self.storage_cl.get_bucket(storage_bucket_name).blob(f'images/{img_file}')
        blob.upload_from_filename(img_file)
        os.remove(f'{img_file}')

        # Caption
        df_new_caption = pd.DataFrame({'PATH': [img_file], 'CAPTION': [caption], 'ID': new_id})
        schema = [SchemaField('PATH', 'STRING'), SchemaField('CAPTION', 'STRING'), SchemaField('CAPTION', 'INTEGER') ]
        self.bigq_cl.insert_rows_from_dataframe(bigquery_table_name, df_new_caption, schema)
