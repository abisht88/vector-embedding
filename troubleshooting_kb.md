# Common Service Errors — Troubleshooting Guide

This document lists common errors seen in production services, their root
causes, and how to fix them.

## 1. Database connection refused

**Error:** `DatabaseConnectionError: Could not connect to database at db.internal:5432: Connection refused`

**Cause:** The database process is down, the host/port is misconfigured, or a
firewall/security group is blocking the connection between the app and the
database.

**Solution:**
- Verify the database service is actually running (`pg_isready` or check the
  DB host's process list).
- Double-check the `DB_HOST` / `DB_PORT` environment variables match the
  actual database endpoint.
- Confirm the security group / firewall rule allows inbound traffic on the
  database port from the app's network.
- If using a connection pool, check it isn't exhausted — raise
  `max_connections` or reduce pool size on the app side.

## 2. Cache node timeout

**Error:** `RedisTimeoutError: Could not reach cache node 10.0.4.12:6379: timed out`

**Cause:** The Redis/cache node is unreachable — either it's down, there's a
network partition, or the client timeout is too aggressive for current
network latency.

**Solution:**
- Check the cache node's health endpoint or `redis-cli ping`.
- Verify security group rules allow the app to reach port 6379.
- Increase the client-side connection/read timeout if latency spikes are
  expected.
- Add a circuit breaker so the app degrades gracefully (skip cache, hit DB
  directly) instead of blocking on a dead cache node.

## 3. Expired auth token

**Error:** `jwt.ExpiredSignatureError: Signature has expired`

**Cause:** The JWT's `exp` claim is in the past — either the token genuinely
expired and wasn't refreshed, or there's clock skew between servers issuing
and validating tokens.

**Solution:**
- Ensure the client refreshes the access token using its refresh token
  before expiry, not just on failure.
- Sync server clocks via NTP — even a few minutes of skew causes spurious
  expirations.
- If tokens expire too quickly for the use case, increase the TTL
  server-side rather than disabling expiry checks.

## 4. Worker killed — out of memory

**Error:** `MemoryError: Worker process killed - out of memory`

**Cause:** The worker's memory usage exceeded its container/process limit,
usually from a memory leak or loading an entire large dataset into memory at
once.

**Solution:**
- Profile the worker for leaks (unbounded caches, unclosed file handles,
  growing in-memory lists).
- Process large datasets in batches/streams instead of loading everything
  at once.
- Raise the container memory limit only as a stopgap — fix the underlying
  allocation pattern first.
- Add memory usage alerting so this is caught before the OOM kill.

## 5. Upstream API rate limited

**Error:** `RateLimitExceededError: 429 Too Many Requests from upstream API`

**Cause:** The service is calling a third-party API faster than its allowed
rate limit.

**Solution:**
- Implement exponential backoff and retry, respecting any `Retry-After`
  header in the response.
- Add a caching layer for repeated identical requests to cut call volume.
- Batch requests where the upstream API supports it.
- If sustained volume genuinely needs more throughput, request a higher
  rate limit from the provider.

## 6. Disk full

**Error:** `OSError: [Errno 28] No space left on device`

**Cause:** The disk/volume backing the service has filled up — usually from
unrotated logs, temp files, or accumulated data that's never cleaned up.

**Solution:**
- Clean up old temp files and rotate logs (e.g. `logrotate`, or cap log
  file size in the logging config).
- Set up disk usage monitoring/alerting well before it hits 100%.
- Increase the volume size if growth is legitimate and expected.
- Audit for anything writing unbounded data (e.g. debug dumps, unpruned
  caches on disk).

## 7. Expired TLS certificate

**Error:** `ssl.SSLCertVerificationError: certificate has expired`

**Cause:** The TLS certificate for the service or an upstream dependency
passed its expiry date without being renewed.

**Solution:**
- Renew the certificate immediately (`certbot renew` or via your cloud
  provider's certificate manager).
- Automate renewal going forward — manual renewal is why this keeps
  happening.
- Add expiry monitoring that alerts 30 days before a cert expires.
