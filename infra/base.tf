data "digitalocean_ssh_key" "operator" {
  name = var.ssh_key_name
}

resource "digitalocean_droplet" "base" {
  name     = "tradan-base"
  size     = "s-2vcpu-2gb"
  image    = "ubuntu-22-04-x64"
  region   = var.region
  vpc_uuid = digitalocean_vpc.tradan.id
  ssh_keys = [data.digitalocean_ssh_key.operator.id]

  user_data = templatefile("${path.module}/scripts/cloud-init-base.yaml", {
    db_password        = var.db_password
    db_name            = var.db_name
    db_user            = var.db_user
    git_repo_url       = local.git_clone_url
    vpc_cidr           = digitalocean_vpc.tradan.ip_range
    pgdata_volume_name   = digitalocean_volume.pgdata.name
    models_volume_name   = digitalocean_volume.models.name
    attach_models_volume = !var.train_enabled
    symbols              = var.symbols
    ingest_retry_enabled = var.ingest_retry_enabled
    db_public_access     = var.db_public_access
    db_worker_ips        = var.db_worker_ips
  })
}

resource "digitalocean_volume_attachment" "pgdata" {
  droplet_id = digitalocean_droplet.base.id
  volume_id  = digitalocean_volume.pgdata.id
}

# When train_enabled=false (distributed worker mode) the models volume lives on base,
# giving workers a persistent rsync target. When train_enabled=true (single train droplet),
# the models volume is attached to the train droplet instead (see train.tf).
resource "digitalocean_volume_attachment" "models_base" {
  count      = var.train_enabled ? 0 : 1
  droplet_id = digitalocean_droplet.base.id
  volume_id  = digitalocean_volume.models.id
}
