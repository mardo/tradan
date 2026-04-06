data "digitalocean_ssh_key" "operator" {
  fingerprint = var.ssh_key_fingerprint
}

resource "digitalocean_droplet" "base" {
  name     = "tradan-base"
  size     = "s-4vcpu-8gb"
  image    = "ubuntu-22-04-x64"
  region   = var.region
  vpc_uuid = digitalocean_vpc.tradan.id
  ssh_keys = [data.digitalocean_ssh_key.operator.id]

  user_data = templatefile("${path.module}/scripts/cloud-init-base.yaml", {
    db_password  = var.db_password
    db_name      = var.db_name
    db_user      = var.db_user
    git_repo_url = var.git_repo_url
  })
}
