package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/rl-sandbox/k8s-cp/config"
	"github.com/rl-sandbox/k8s-cp/handler"
	"github.com/rl-sandbox/k8s-cp/k8s"
	"github.com/rl-sandbox/k8s-cp/reconciler"
)

func main() {
	// Load configuration
	cfg := config.LoadConfig()

	// Create Redis client
	rdb := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", cfg.RedisHost, cfg.RedisPort),
		DB:   cfg.RedisDB,
	})
	defer rdb.Close()

	// Ping Redis to ensure connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	// Create Kubernetes client
	clientset, err := k8s.GetK8sClient()
	if err != nil {
		log.Fatalf("Failed to create Kubernetes client: %v", err)
	}
	log.Println("Kubernetes client initialized successfully")

	// Set up Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	// Health check endpoints
	r.GET("/healthz", func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	})

	r.GET("/readyz", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 500*time.Millisecond)
		defer cancel()

		if err := rdb.Ping(ctx).Err(); err != nil {
			c.String(http.StatusServiceUnavailable, "redis not ready")
			return
		}

		c.String(http.StatusOK, "ready")
	})

	// Metrics endpoint
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Create handler dependencies
	deps := handler.Dependencies{
		Clientset: clientset,
		Redis:     rdb,
		Config:    cfg,
	}

	// Main API endpoints
	r.POST("/create", handler.HandleSpawn(deps))
	r.DELETE("/destroy", handler.HandleDestroy(deps))

	// Create HTTP server
	srv := http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	// Start reconciler in background
	reconcilerCtx, reconcilerCancel := context.WithCancel(context.Background())
	defer reconcilerCancel()
	go reconciler.Start(reconcilerCtx, clientset, rdb, cfg)

	// Start server in a goroutine
	go func() {
		log.Println("Starting control-plane server on :8080")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Stop reconciler
	reconcilerCancel()

	// Create shutdown context with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	// Shutdown the server
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited properly")
}
