resource "digitalocean_droplet" "train" {
  count    = var.train_enabled ? 1 : 0
  name     = "tradan-train"
  size     = var.train_droplet_size
  image    = "ubuntu-22-04-x64"
  region   = var.region
  vpc_uuid = digitalocean_vpc.tradan.id
  ssh_keys = [data.digitalocean_ssh_key.operator.id]

  user_data = templatefile("${path.module}/scripts/cloud-init-train.yaml", {
    db_host      = digitalocean_droplet.base.ipv4_address_private
    db_password  = var.db_password
    db_name      = var.db_name
    db_user      = var.db_user
    worker_count = local.worker_count
  })
}

resource "digitalocean_volume_attachment" "models" {
  count      = var.train_enabled ? 1 : 0
  droplet_id = digitalocean_droplet.train[0].id
  volume_id  = digitalocean_volume.models.id
}
