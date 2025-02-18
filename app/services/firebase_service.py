import firebase_admin
from firebase_admin import credentials, firestore
from app.core.config import settings

# Initialize Firebase
cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)

db = firestore.client()

def save_prediction(user_id: str, result: dict):
    """Save prediction in Firebase Firestore"""
    doc_ref = db.collection('predictions').add({
        'user_id': user_id,
        'result': result,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    return doc_ref