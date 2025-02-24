from sagemaker.model import Model
import sagemaker

# Initialize SageMaker session in us-east-1
sagemaker_session = sagemaker.Session(boto_session=sagemaker.Session().boto_session)

# Use the correct execution role (replace with your ARN)
role = "arn:aws:iam::904233093112:role/SageMakerExecutionRole"

# Use the correct ECR image in us-east-1
container_uri = "904233093112.dkr.ecr.us-east-1.amazonaws.com/wine-quality:latest"

# Deploy the model in SageMaker
model = Model(
    image_uri=container_uri,
    role=role,
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    endpoint_name="wine-quality-endpoint",
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("Model deployed successfully!")
