#!/usr/bin/env python3
"""
Deploy microSAM service to BioEngine worker.
"""
import asyncio
import os
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

async def deploy_microsam():
    print("Connecting to BioEngine worker...")
    
    # Load .env file if it exists
    load_dotenv()
    
    # Use HYPHA_TOKEN from environment if available, otherwise login
    token = os.getenv("HYPHA_TOKEN") or await login({"server_url": "https://hypha.aicell.io"})
    # Connect to BioEngine worker
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
    })
    # Songtao's workspace
    workspace = "agent-lens"
    # Get the worker service
    worker = await server.get_service(f"{workspace}/bioengine-worker")
    
    print("Loading microSAM service files...")
    
    # Load service files - BioEngine expects files with type and content
    files = [
        {
            "name": "micro-sam.py",
            "type": "text",
            "content": open("bioengine-app/micro-sam.py", "r").read()
        },
        {
            "name": "manifest.yaml", 
            "type": "text",
            "content": open("bioengine-app/manifest.yaml", "r").read()
        }
    ]
    
    print("Deploying microSAM service...")
    
    # Deploy the service
    await worker.save_application(files)
    
    # Run application with token if available
    app_params = {"artifact_id": "micro-sam", "application_id": "micro-sam"}
    if os.getenv("HYPHA_TOKEN"):
        app_params["hypha_token"] = os.getenv("HYPHA_TOKEN")
    
    await worker.run_application(**app_params)
    
    print("✅ microSAM service deployed successfully!")
    
    # Wait for the service to be deployed and started
    print("Waiting for service to start up...")
    await asyncio.sleep(30)  # Wait longer for Ray Serve to start
    
    # Get the actual service name from the deployment
    print("Getting deployed service name...")
    applications = await worker.list_applications()
    print(f"Available applications: {list(applications.keys())}")
    
    # Connect directly to the deployed micro-sam application
    print("Connecting to deployed microSAM application...")
    
    try:
        # Find the micro-sam application (it might have a prefix like 'agent-lens/')
        microsam_app_name = None
        for app_name in applications.keys():
            if app_name.endswith('/micro-sam') or app_name == 'micro-sam':
                microsam_app_name = app_name
                break
        
        if not microsam_app_name:
            print("❌ micro-sam application not found in deployed applications")
            print(f"Available applications: {list(applications.keys())}")
            return
        
        print(f"✅ Found microSAM application: {microsam_app_name}")
        
        # Get the service from the application
        print("Getting microSAM service...")
        microsam = await server.get_service(microsam_app_name)
        
        # Test the service with retry logic
        print("\nTesting service...")
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                
                # Check if service is working
                status = await microsam.get_fit_status()
                print(f"✅ Service is working! Status: {status['status']}")
                
                break
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ All attempts failed. Service may need more time to start.")
                    print("You can try connecting to the service manually later.")
                    
    except Exception as e:
        print(f"❌ Error connecting to microSAM service: {e}")
        print("The application was deployed but the service may not be running yet.")
    
if __name__ == "__main__":
    asyncio.run(deploy_microsam())
