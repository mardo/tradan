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
    pgdata_volume_name = digitalocean_volume.pgdata.name
    symbols              = var.symbols
    ingest_retry_enabled = var.ingest_retry_enabled
  })
}

resource "digitalocean_volume_attachment" "pgdata" {
  droplet_id = digitalocean_droplet.base.id
  volume_id  = digitalocean_volume.pgdata.id
}
