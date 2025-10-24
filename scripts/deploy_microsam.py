#!/usr/bin/env python3
"""
Deploy microSAM service to BioEngine worker.
"""
import asyncio
from hypha_rpc import connect_to_server, login

async def deploy_microsam():
    print("Connecting to BioEngine worker...")
    
    token = await login({"server_url": "https://hypha.aicell.io"})
    # Connect to BioEngine worker
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token,
    })
    # Songtao's workspace
    workspace = "ws-user-google-oauth2|103047988474094226050"
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
    await worker.run_application("micro-sam")
    
    print("✅ microSAM service deployed successfully!")
    
    # Wait for the service to be deployed and started
    print("Waiting for service to start up...")
    await asyncio.sleep(30)  # Wait longer for Ray Serve to start
    
    # Get the actual service name from the deployment
    print("Getting deployed service name...")
    applications = await worker.list_applications()
    print(f"Available applications: {list(applications.keys())}")
    
    # Find the micro-sam application
    # applications is a dict where keys are service names
    microsam_service_name = None
    for service_name in applications.keys():
        if 'micro-sam' in service_name.lower():
            microsam_service_name = service_name
            break
    
    if microsam_service_name:
        print(f"✅ Found microSAM application: {microsam_service_name}")
        
        # Get the actual running services from the application
        print("Getting running services...")
        try:
            # List all services to find the actual running service
            all_services = await server.list_services()
            print(f"All available services: {all_services}")
            
            # Look for the actual microSAM service by testing functions
            # BioEngine generates random service IDs, so we need to find the one that contains our functions
            microsam_running_service = None
            
            # Filter out obvious non-microSAM services to avoid timeouts
            microsam_candidates = []
            for service_info in all_services:
                # Extract the service ID from the service info object
                if isinstance(service_info, dict):
                    service_id = service_info.get('id', service_info.get('name', str(service_info)))
                else:
                    service_id = str(service_info)
                
                # Skip obvious non-microSAM services to avoid timeouts
                if any(skip_term in service_id.lower() for skip_term in ['rtc', 'built-in', 'webrtc', 'proxy']):
                    print(f"Skipping non-microSAM service: {service_id}")
                    continue
                    
                microsam_candidates.append(service_id)
            
            print(f"Testing {len(microsam_candidates)} potential microSAM services...")
            
            for service_id in microsam_candidates:
                # Check if this service has the microSAM functions we expect
                try:
                    print(f"Testing service: {service_id}")
                    # Use a shorter timeout to avoid long waits
                    test_service = await server.get_service(service_id)
                    # Try to call a microSAM-specific function to verify it's our service
                    await test_service.get_fit_status()
                    microsam_running_service = service_id
                    print(f"✅ Found microSAM service by testing functions: {service_id}")
                    break
                except Exception as e:
                    # This service doesn't have our functions, continue searching
                    print(f"  Not microSAM service: {str(e)[:100]}...")
                    continue
            
            if microsam_running_service:
                print(f"✅ Found running microSAM service: {microsam_running_service}")
                
                # Test the service with retry logic
                print("\nTesting service...")
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1}/{max_retries}...")
                        microsam = await server.get_service(microsam_running_service)
                        
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
            else:
                print("⚠️  Could not find the running microSAM service")
                print("The application was deployed but the service may not be running yet.")
        except Exception as e:
            print(f"⚠️  Error getting services: {e}")
    else:
        print("⚠️  Could not find deployed microSAM application")
    
if __name__ == "__main__":
    asyncio.run(deploy_microsam())
