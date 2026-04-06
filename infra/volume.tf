resource "digitalocean_volume" "models" {
  name                    = "tradan-models"
  region                  = var.region
  size                    = 100
  initial_filesystem_type = "ext4"
  description             = "Persistent storage for trained model .zip files"
}
