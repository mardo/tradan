terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
  required_version = ">= 1.6"
}

provider "digitalocean" {
  token = var.do_token
}

locals {
  worker_counts = {
    "c-16" = 14
    "c-32" = 28
    "c-48" = 44
  }
  worker_count = local.worker_counts[var.train_droplet_size]
}

resource "digitalocean_vpc" "tradan" {
  name     = "tradan-vpc"
  region   = var.region
  ip_range = "10.0.0.0/16"
}

resource "digitalocean_firewall" "tradan" {
  name = "tradan-firewall"

  droplet_ids = concat(
    [digitalocean_droplet.base.id],
    var.train_enabled ? [digitalocean_droplet.train[0].id] : []
  )

  # SSH: only from operator IP
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = [var.operator_ip]
  }

  # PostgreSQL: only from within VPC
  inbound_rule {
    protocol         = "tcp"
    port_range       = "5432"
    source_addresses = [digitalocean_vpc.tradan.ip_range]
  }

  # Allow all outbound (package installs, git clone, etc.)
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
