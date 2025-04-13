import oci

# Initialize the client
config = oci.config.from_file("/.oci/config")  # Your OCI config file
object_storage = oci.object_storage.ObjectStorageClient(config)

# Specify bucket and namespace
namespace = object_storage.get_namespace().data
bucket_name = "your_bucket_name"

# Create a pre-authenticated request
par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
    name="example-par",
    access_type="ObjectRead",  # Can be 'ObjectRead', 'ObjectWrite', or 'ObjectReadWrite'
    object_name="object-name",  # Optional: leave blank for bucket-level access
    time_expires="2025-04-20T00:00:00Z"  # Expiry date and time)
)
response = object_storage.create_preauthenticated_request(
    namespace_name=namespace,
    bucket_name=bucket_name,
    create_preauthenticated_request_details=par_details
)

par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
print(f"Pre-Authenticated Request URL: {response.data.access_uri}")