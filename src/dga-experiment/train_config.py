from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails
from azureml.core import Workspace
from azureml.core import Dataset
import azureml.core
import os
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
ws.set_default_datastore('dgadatastore')
default_ds = ws.get_default_datastore()
env = Environment.from_conda_specification("experiment_env", "dga-experiment/environment.yml")
malicious_dga = ws.datasets.get("dga")
clean_domains = ws.datasets.get("alexadomains")

# Choose a name for your cluster.
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           max_nodes=4)

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())

script_config = ScriptRunConfig(source_directory='dga-experiment',
                              script='train.py',
                              arguments = ['--regularization', 0.1, # Regularizaton rate parameter
                                           '--input-data-dga', malicious_dga.as_named_input('dga'),
                                           '--input-data-clean', clean_domains.as_named_input('clean_domains')], # Reference to dataset
                              environment=env,
                              compute_target=compute_target,
                              ) 
                            #   docker_runtime_config=DockerConfiguration(use_docker=True)

# submit the experiment
experiment_name = 'malicious-dga'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
run.wait_for_completion()