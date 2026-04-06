variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_fingerprint" {
  description = "Fingerprint of the SSH key registered in DigitalOcean"
  type        = string
}

variable "operator_ip" {
  description = "Your IP in CIDR notation for SSH firewall rule (e.g. 1.2.3.4/32). Defaults to 0.0.0.0/0 (open to world — safe because SSH is key-only). The Makefile auto-detects and passes your current IP on every apply."
  type        = string
  default     = "0.0.0.0/0"
}

variable "db_password" {
  description = "PostgreSQL password for the tradan user"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "tradan"
}

variable "db_user" {
  description = "PostgreSQL user"
  type        = string
  default     = "tradan"
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "nyc3"
}

variable "train_enabled" {
  description = "Set to true to create the training droplet, false to destroy it"
  type        = bool
  default     = false
}

variable "train_droplet_size" {
  description = "CPU-optimized droplet size: c-16 (14 workers), c-32 (28 workers), c-48 (44 workers)"
  type        = string
  default     = "c-32"

  validation {
    condition     = contains(["c-16", "c-32", "c-48"], var.train_droplet_size)
    error_message = "train_droplet_size must be c-16, c-32, or c-48."
  }
}

variable "git_repo_url" {
  description = "Git repository URL to clone on droplets (e.g. git@github.com:org/tradan.git)"
  type        = string
}
