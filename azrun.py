import azureml
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import DockerConfiguration
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import ScriptRunConfig


ia = InteractiveLoginAuthentication(tenant_id='dce87315-8ffa-4a01-ab40-0de5a7214b2f') #Add yours
ws = Workspace(subscription_id= "npunext-1680261565427", #Add yours
    resource_group="devop-demo-2603", #Add yours
    workspace_name= "harish-ws", auth=ia) #Add yours
print(f'worspace details {ws}')
cluster_name = 'compute-cluster'
try:
 compute_target = ComputeTarget(workspace=ws, name=cluster_name)
 print('Found existing coumpute target')
except ComputeTargetException:
 print('Createing a new compute target...')
 compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D2as_v4', max_nodes=2)
 compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
 compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
print('-'*101)
print('Compute target created')
print('-'*101)
demo_env = Environment(name="") #Add yours
print('loading the conda dependencies..')
for pip_package in ["joblib","scikit-learn","pandas","azureml-sdk", "numpy"]:
    demo_env.python.conda_dependencies.add_pip_package(pip_package)
demo_env.register(workspace=ws)
print('-'*101)
print('Environment variables created.')
print('-'*101)
docker_config = DockerConfiguration(use_docker=True)
print('running the script my friend.')
src = ScriptRunConfig(source_directory='.',
      script='train.py',
      compute_target=compute_target,
      environment=demo_env,
      docker_runtime_config= docker_config
      )
print('Completed running the script.')
run = Experiment(workspace=ws, name='').submit(src) #Add yours
run.wait_for_completion(show_output=True)
model_name = 'random_forest_regression_model.pkl'
print('Registering the model.')
if run.get_status() == 'Completed':
 model = run.register_model(
      model_name=model_name,
         model_path=f'outputs/{model_name}'
        )
 print('Model registered!')
print('Experiment completed.')