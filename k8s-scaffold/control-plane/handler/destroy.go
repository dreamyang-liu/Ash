package handler

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/rl-sandbox/k8s-cp/k8s"
	"github.com/rl-sandbox/k8s-cp/metrics"
	"github.com/rl-sandbox/k8s-cp/store"
)

// HandleDestroy handles DELETE /destroy requests
func HandleDestroy(deps Dependencies) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			IDs []string `json:"ids"`
			All bool     `json:"all"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if !req.All && len(req.IDs) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "must specify 'ids' or 'all: true'"})
			return
		}

		ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Minute)
		defer cancel()

		var succeeded []string
		var failed []string

		if req.All {
			// Destroy all sandboxes
			succeeded, failed = destroyAll(ctx, deps)
		} else {
			// Destroy specific sandboxes by UUID
			succeeded, failed = destroyByIDs(ctx, deps, req.IDs)
		}

		log.Printf("Destroy completed: succeeded=%d failed=%d", len(succeeded), len(failed))
		c.JSON(http.StatusOK, gin.H{
			"deleted": succeeded,
			"failed":  failed,
			"count":   len(succeeded),
		})
	}
}

// destroyAll destroys all sandboxes
func destroyAll(ctx context.Context, deps Dependencies) ([]string, []string) {
	var succeeded []string
	var failed []string

	selector := "from=control-plane,type=sandbox"
	deployments, err := deps.Clientset.AppsV1().Deployments(deps.Config.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})
	if err != nil {
		log.Printf("Failed to list deployments: %v", err)
		return succeeded, failed
	}

	for _, dep := range deployments.Items {
		name := dep.Name
		namespace := dep.Namespace
		id := fmt.Sprintf("%s/%s", namespace, name)

		if err := k8s.DestroySandbox(ctx, deps.Clientset, namespace, name); err != nil {
			log.Printf("Failed to destroy sandbox %s: %v", id, err)
			failed = append(failed, id)
			metrics.SandboxDestroyTotal.WithLabelValues("error").Inc()
			continue
		}

		// Delete Redis records
		pattern := fmt.Sprintf("sandbox:%s-*", name)
		if err := store.ScanAndDelete(ctx, deps.Redis, pattern); err != nil {
			log.Printf("Failed to delete Redis records for %s: %v", id, err)
			failed = append(failed, id)
		} else {
			succeeded = append(succeeded, id)
			metrics.SandboxDestroyTotal.WithLabelValues("success").Inc()
		}
	}

	return succeeded, failed
}

// destroyByIDs destroys specific sandboxes by UUID
func destroyByIDs(ctx context.Context, deps Dependencies, uuids []string) ([]string, []string) {
	var succeeded []string
	var failed []string

	for _, uuid := range uuids {
		result, err := store.GetRecord(ctx, deps.Redis, uuid)
		if err != nil || len(result) == 0 {
			log.Printf("Destroy failed: UUID %s not found", uuid)
			failed = append(failed, uuid)
			metrics.SandboxDestroyTotal.WithLabelValues("error").Inc()
			continue
		}

		// Use name/namespace fields if available, fall back to hostname parsing
		svcName := result["name"]
		namespace := result["namespace"]

		if svcName == "" || namespace == "" {
			// Backwards compatibility: parse from hostname
			host := result["host"]
			parts := strings.Split(host, ".")
			if len(parts) < 2 {
				log.Printf("Destroy failed: invalid host format for UUID %s", uuid)
				failed = append(failed, uuid)
				metrics.SandboxDestroyTotal.WithLabelValues("error").Inc()
				continue
			}
			svcName = parts[0]
			namespace = parts[1]
		}

		if err := k8s.DestroySandbox(ctx, deps.Clientset, namespace, svcName); err != nil {
			log.Printf("Failed to destroy sandbox %s: %v", uuid, err)
		}

		if err := store.DeleteRecord(ctx, deps.Redis, uuid); err != nil {
			log.Printf("Failed to delete Redis key for %s: %v", uuid, err)
		}

		succeeded = append(succeeded, uuid)
		metrics.SandboxDestroyTotal.WithLabelValues("success").Inc()
	}

	return succeeded, failed
}
