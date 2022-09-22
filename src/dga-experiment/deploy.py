from azureml.core.model import InferenceConfig
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice
from azureml.core.model import Model
from azureml.core.compute_target import ComputeTargetException


def run_deploy():

    ws = Workspace.from_config()
    inference_config = InferenceConfig(runtime= "python",
                                                source_directory = 'dga-experiment',
                                                entry_script="scoring.py",
                                                conda_file="environment.yml")
    
    # Create new AKS Cluster if it does not exist, otherwise load
    cluster_name = 'aks-cluster'
    try:
        production_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    except ComputeTargetException:
        compute_config = AksCompute.provisioning_configuration(location='eastus')
        production_cluster = ComputeTarget.create(
            workspace=ws, name=cluster_name, provisioning_configuration=compute_config
        )
        production_cluster.wait_for_completion(show_output=True)
                                                
    #Deployment configuration
    deploy_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
                                                                
    print('Deploying model...')
    # model_json = ws.models['elmo-model.json']
    model_json = Model(ws,'elmo-model.json',version=7)
    # model_weights = ws.models['elmo-model-weights.h5']
    model_weights = Model(ws,'elmo-model-weights.h5',version=7)
    # ws.models['elmo-model.json:8']

    service = Model.deploy(workspace=ws,
                        name = 'elmo-service',
                        models = [model_json,model_weights],
                        inference_config = inference_config,
                        deployment_config = deploy_config,
                        deployment_target = production_cluster,
                        overwrite=True)
    service.wait_for_deployment(show_output = True)
    
    # Service details
    print(service.state)
    print(service.get_logs())
    endpoint = service.scoring_uri
    print(endpoint)

if __name__=='__main__':
    print("Call run_config")
    run_deploy()