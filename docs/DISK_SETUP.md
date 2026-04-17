# Disk Setup

This guide covers mounting a secondary disk to `/data` on a fresh Ubuntu 24.04
instance. Follow this before cloning the repo if you are using a separate disk
volume for your data directory.

If you are using the boot disk only, skip this — `/data` will be created
automatically by `setup.sh`.

---

## 1. Create the mount point

```bash
sudo mkdir /data
```

---

## 2. Identify the disk

```bash
lsblk
```

You should see two block devices — your boot disk and your data disk. Example:

```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
vda    252:0    0   50G  0 disk
└─vda1 252:1    0   50G  0 part /
vdc    252:32   0 1280G  0 disk
```

The data disk (`vdc` in this example) has no mountpoint — that's the one to mount.

---

## 3. Check the disk

```bash
sudo file -s /dev/vdc
```

If the output is `/dev/vdc: data` the disk is unformatted — proceed to step 4.
If it shows a filesystem type the disk already has data on it — skip formatting.

---

## 4. Format the disk

> **Only do this on a fresh disk. This destroys all existing data.**

```bash
sudo mkfs.ext4 /dev/vdc
```

Replace `vdc` with your actual device name from step 2.

---

## 5. Mount the disk

```bash
sudo mount /dev/vdc /data
```

---

## 6. Set ownership

```bash
sudo chown $USER:$USER /data
```

---

## 7. Persist the mount across reboots

Get the disk's UUID:

```bash
sudo blkid /dev/vdc
```

Output will look like:

```
/dev/vdc: UUID="f161f289-ea2c-478d-960a-ead94e9142e5" TYPE="ext4"
```

Add an entry to `/etc/fstab`:

```bash
echo 'UUID=your-uuid-here  /data  ext4  defaults  0  2' | sudo tee -a /etc/fstab
```

Replace `your-uuid-here` with your actual UUID from the `blkid` output.

Verify the fstab entry is correct:

```bash
sudo mount -a
```

No output means success. If you see errors, check the UUID in `/etc/fstab`.

---

## 8. Verify

```bash
df -h /data
```

Expected output:

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/vdc        1.3T   28K  1.2T   1% /data
```

`/data` is now ready. Return to the [README](../README.md) and proceed with
the clone step.