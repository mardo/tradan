output "base_ip" {
  description = "Public IP of the base droplet"
  value       = digitalocean_droplet.base.ipv4_address
}

output "base_private_ip" {
  description = "Private VPC IP of the base droplet (used by training droplet for DB)"
  value       = digitalocean_droplet.base.ipv4_address_private
}

output "base_ssh" {
  description = "SSH command to connect to the base droplet"
  value       = "ssh root@${digitalocean_droplet.base.ipv4_address}"
}

output "train_ip" {
  description = "Public IP of the training droplet (empty if not running)"
  value       = var.train_enabled ? digitalocean_droplet.train[0].ipv4_address : ""
}

output "train_ssh" {
  description = "SSH command to connect to the training droplet"
  value       = var.train_enabled ? "ssh root@${digitalocean_droplet.train[0].ipv4_address}" : "training droplet is not running"
}

output "worker_count" {
  description = "Number of parallel training workers for the current droplet size"
  value       = local.worker_count
}

output "volume_id" {
  description = "Block volume ID (reference for manual operations)"
  value       = digitalocean_volume.models.id
}

output "droplet_size" {
  description = "Current training droplet size"
  value       = var.train_droplet_size
}
