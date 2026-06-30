package reconciler

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/go-redis/redis/v8"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/rl-sandbox/k8s-cp/config"
	"github.com/rl-sandbox/k8s-cp/metrics"
	"github.com/rl-sandbox/k8s-cp/store"
)

// Start begins the reconciliation loop
func Start(ctx context.Context, clientset *kubernetes.Clientset, rdb *redis.Client, cfg *config.Config) {
	interval := time.Duration(cfg.ReconcileInterval) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	log.Printf("Starting reconciler with interval %s", interval)

	// Run immediately on startup
	reconcile(ctx, clientset, rdb, cfg)

	for {
		select {
		case <-ctx.Done():
			log.Println("Reconciler stopped")
			return
		case <-ticker.C:
			reconcile(ctx, clientset, rdb, cfg)
		}
	}
}

func reconcile(ctx context.Context, clientset *kubernetes.Clientset, rdb *redis.Client, cfg *config.Config) {
	log.Println("Running reconciliation...")

	// Phase 1: Sync K8s -> Redis (self-healing: create missing Redis records)
	if err := syncK8sToRedis(ctx, clientset, rdb, cfg); err != nil {
		log.Printf("Failed to sync K8s to Redis: %v", err)
	}

	// Phase 2: Sync Redis -> K8s (cleanup: delete stale Redis records)
	if err := syncRedisToK8s(ctx, clientset, rdb, cfg); err != nil {
		log.Printf("Failed to sync Redis to K8s: %v", err)
	}

	// Update active count metric
	updateActiveCount(ctx, rdb)
}

// syncK8sToRedis ensures all K8s deployments have Redis records
func syncK8sToRedis(ctx context.Context, clientset *kubernetes.Clientset, rdb *redis.Client, cfg *config.Config) error {
	selector := "from=control-plane,type=sandbox"
	deployments, err := clientset.AppsV1().Deployments(cfg.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})
	if err != nil {
		return fmt.Errorf("failed to list deployments: %w", err)
	}

	for _, dep := range deployments.Items {
		name := dep.Name
		namespace := dep.Namespace

		// Check if Redis record exists (scan for keys matching pattern)
		pattern := fmt.Sprintf("sandbox:%s-*", name)
		iter := rdb.Scan(ctx, 0, pattern, 10).Iterator()
		found := false
		for iter.Next(ctx) {
			found = true
			break
		}
		if iter.Err() != nil {
			log.Printf("Failed to scan for %s: %v", pattern, iter.Err())
			continue
		}

		if !found {
			log.Printf("Reconciler: Creating missing Redis record for deployment %s/%s", namespace, name)

			// Get service to fetch port info
			svc, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
			port := 3000
			if err == nil && len(svc.Spec.Ports) > 0 {
				port = int(svc.Spec.Ports[0].Port)
			}

			// Create Redis record with a generated UUID
			// Note: We can't recover the original UUID, so we create a new one
			uuid := fmt.Sprintf("%s-reconciled-%d", name, time.Now().Unix())
			record := store.SandboxRecord{
				UUID:      uuid,
				Host:      fmt.Sprintf("%s.%s.svc.cluster.local", name, namespace),
				Port:      port,
				Status:    "ready",
				Name:      name,
				Namespace: namespace,
			}

			if err := store.SaveRecord(ctx, rdb, record); err != nil {
				log.Printf("Failed to save reconciled record: %v", err)
			} else {
				log.Printf("Reconciler: Created Redis record for %s with UUID %s", name, uuid)
			}
		}
	}

	return nil
}

// syncRedisToK8s removes Redis records for deleted K8s deployments
func syncRedisToK8s(ctx context.Context, clientset *kubernetes.Clientset, rdb *redis.Client, cfg *config.Config) error {
	records, err := store.ListAllRecords(ctx, rdb)
	if err != nil {
		return fmt.Errorf("failed to list Redis records: %w", err)
	}

	for _, rec := range records {
		name := rec["name"]
		namespace := rec["namespace"]
		uuid := rec["uuid"]

		// Fallback to parsing host if name/namespace are missing (backwards compat)
		if name == "" || namespace == "" {
			host := rec["host"]
			// Try to extract from host (e.g., "service.namespace.svc.cluster.local")
			if host != "" {
				parts := splitHostname(host)
				if len(parts) >= 2 {
					name = parts[0]
					namespace = parts[1]
				}
			}
		}

		if name == "" || namespace == "" {
			log.Printf("Reconciler: Skipping record with missing name/namespace: %v", rec)
			continue
		}

		// Check if deployment exists
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			log.Printf("Reconciler: Deployment %s/%s no longer exists, deleting Redis record %s", namespace, name, uuid)
			if err := store.DeleteRecord(ctx, rdb, uuid); err != nil {
				log.Printf("Failed to delete stale record %s: %v", uuid, err)
			}
		}
	}

	return nil
}

// updateActiveCount updates the active sandbox count metric
func updateActiveCount(ctx context.Context, rdb *redis.Client) {
	records, err := store.ListAllRecords(ctx, rdb)
	if err != nil {
		log.Printf("Failed to count active sandboxes: %v", err)
		return
	}

	metrics.SandboxActiveCount.Set(float64(len(records)))
}

// splitHostname splits a hostname like "service.namespace.svc.cluster.local"
func splitHostname(host string) []string {
	var result []string
	start := 0
	for i := 0; i < len(host); i++ {
		if host[i] == '.' {
			result = append(result, host[start:i])
			start = i + 1
		}
	}
	if start < len(host) {
		result = append(result, host[start:])
	}
	return result
}
