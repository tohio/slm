# Disk Setup

This guide covers mounting a secondary disk to `/data` on a fresh Ubuntu 24.04
instance. Follow this before cloning the repo if you are using a separate disk
volume for your data directory.

If you are using the boot disk only, skip this — `/data` will be created
automatically by `setup.sh`.

---

## 1. Identify the disk

```bash
lsblk
```

You should see two block devices — your boot disk (usually `sda` or `nvme0n1`)
and your data disk (usually `sdb`, `nvme1n1`, or similar). Example output:

```
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
sda           8:0    0    50G  0 disk
└─sda1        8:1    0    50G  0 part /
nvme1n1     259:0    0  1280G  0 disk
```

The data disk (`nvme1n1` in this example) has no mountpoint — that's the one
to mount.

---

## 2. Format the disk

> **Only do this on a fresh disk. This destroys all existing data.**

```bash
sudo mkfs.ext4 /dev/nvme1n1
```

Replace `nvme1n1` with your actual device name from step 1.

---

## 3. Mount the disk

```bash
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data
```

---

## 4. Persist the mount across reboots

Get the disk's UUID:

```bash
sudo blkid /dev/nvme1n1
```

Output will look like:
```
/dev/nvme1n1: UUID="a1b2c3d4-e5f6-..." TYPE="ext4"
```

Add an entry to `/etc/fstab`:

```bash
echo "UUID=a1b2c3d4-e5f6-...  /data  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab
```

Replace the UUID with your actual value. The `nofail` option ensures the
instance still boots if the disk is detached.

Verify the fstab entry is correct:

```bash
sudo mount -a
```

No output means success. If you see errors, check the UUID in `/etc/fstab`.

---

## 5. Set ownership

```bash
sudo chown -R $USER:$USER /data
```

---

## 6. Verify

```bash
df -h /data
```

Expected output:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1    1.3T   28K  1.2T   1% /data
```

`/data` is now ready. Return to the [README](../README.md) and proceed with
the clone step.