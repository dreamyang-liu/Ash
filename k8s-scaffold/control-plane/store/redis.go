package store

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/rl-sandbox/k8s-cp/metrics"
)

// SandboxRecord represents a sandbox record in Redis
type SandboxRecord struct {
	UUID      string
	Host      string
	Port      int
	Status    string
	Name      string
	Namespace string
	Path      string // endpoint path, e.g. "/mcp" or "/jsonrpc"
}

// SaveRecord saves a sandbox record to Redis
func SaveRecord(ctx context.Context, rdb *redis.Client, record SandboxRecord) error {
	start := time.Now()
	defer func() {
		metrics.RedisOpDuration.WithLabelValues("hset").Observe(time.Since(start).Seconds())
	}()

	key := fmt.Sprintf("sandbox:%s", record.UUID)
	data := map[string]interface{}{
		"uuid":      record.UUID,
		"host":      record.Host,
		"port":      record.Port,
		"status":    record.Status,
		"name":      record.Name,
		"namespace": record.Namespace,
		"path":      record.Path,
	}

	pipe := rdb.Pipeline()
	pipe.HSet(ctx, key, data)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to save sandbox record: %w", err)
	}

	return nil
}

// GetRecord retrieves a sandbox record from Redis
func GetRecord(ctx context.Context, rdb *redis.Client, uuid string) (map[string]string, error) {
	start := time.Now()
	defer func() {
		metrics.RedisOpDuration.WithLabelValues("hgetall").Observe(time.Since(start).Seconds())
	}()

	key := fmt.Sprintf("sandbox:%s", uuid)
	result, err := rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get sandbox record: %w", err)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("sandbox not found")
	}

	return result, nil
}

// DeleteRecord deletes a sandbox record from Redis
func DeleteRecord(ctx context.Context, rdb *redis.Client, uuid string) error {
	start := time.Now()
	defer func() {
		metrics.RedisOpDuration.WithLabelValues("del").Observe(time.Since(start).Seconds())
	}()

	key := fmt.Sprintf("sandbox:%s", uuid)
	if err := rdb.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("failed to delete sandbox record: %w", err)
	}

	return nil
}

// ScanAndDelete scans for keys matching a pattern and deletes them
func ScanAndDelete(ctx context.Context, rdb *redis.Client, pattern string) error {
	start := time.Now()
	defer func() {
		metrics.RedisOpDuration.WithLabelValues("scan").Observe(time.Since(start).Seconds())
	}()

	iter := rdb.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		if err := rdb.Del(ctx, iter.Val()).Err(); err != nil {
			log.Printf("Failed to delete key %s: %v", iter.Val(), err)
		}
	}

	if err := iter.Err(); err != nil {
		return fmt.Errorf("scan error: %w", err)
	}

	return nil
}

// ListAllRecords lists all sandbox records from Redis
func ListAllRecords(ctx context.Context, rdb *redis.Client) ([]map[string]string, error) {
	start := time.Now()
	defer func() {
		metrics.RedisOpDuration.WithLabelValues("scan").Observe(time.Since(start).Seconds())
	}()

	var records []map[string]string
	iter := rdb.Scan(ctx, 0, "sandbox:*", 0).Iterator()
	for iter.Next(ctx) {
		result, err := rdb.HGetAll(ctx, iter.Val()).Result()
		if err != nil {
			log.Printf("Failed to get record %s: %v", iter.Val(), err)
			continue
		}
		if len(result) > 0 {
			records = append(records, result)
		}
	}

	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("scan error: %w", err)
	}

	return records, nil
}
