# Basic Setup for the backend locally

1. Create a virtual environment: python -m venv .venv
2. .venv\Scripts\Activate.ps1 - You will have to do this every time you want to run this. Windows: .venv\Scripts\Activate.ps1, Linux/Mac: source .venv/bin/activate
3. (Optional)  python -m pip install --upgrade pip
4. pip install -r requirements.txt
5. Go here: https://console.firebase.google.com/project/pantry-ai-db083/settings/serviceaccounts/adminsdk and Generate a new private key and drag the json file into this folder (not app). *Make sure that it is named service-account.json and that it will not be committed (it is in the gitignore)*. Then create a .env file and add a field called FIREBASE_API_KEY. Then go to firebase, click on 1 app, then on the settings cog, and scroll down to the firebaseConfig object. You will see apiKey as one of the fields. Similar thing here, this should not be committed. 
6. To run dev (good for using insomnia/postman) - fastapi dev app/main.py. 
If you want to run and get it working with the mobile app, do fastapi run app/main.py which will host it on port 8000. To get the ip on windows go to ipconfig on Windows. 
7. When you are done, Ctrl+c and then type deactivate to deactivate the venv. 

# Basic setup to host on Google Cloud run 
1. Make sure that the two for prod lines are uncommented and the two local dev lines are commented out in main.py (change this back when committing)
2. gcloud builds submit --tag gcr.io/pantry-ai-db083/backend
3. gcloud run deploy --image gcr.io/pantry-ai-db083/backend-new --platform managed --allow-unauthenticated --execution-environment gen1 --max-instances 5 --cpu 1 --memory 1Gi         (specs can be changed as needed). *Do not set min-instances and especially don't set it to something >0 for the time being!*       
