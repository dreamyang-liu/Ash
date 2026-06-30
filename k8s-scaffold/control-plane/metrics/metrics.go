package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// SandboxCreateTotal counts sandbox creation attempts
	SandboxCreateTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sandbox_create_total",
			Help: "Total number of sandbox creation attempts",
		},
		[]string{"status"},
	)

	// SandboxDestroyTotal counts sandbox destruction attempts
	SandboxDestroyTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sandbox_destroy_total",
			Help: "Total number of sandbox destruction attempts",
		},
		[]string{"status"},
	)

	// SandboxCreateDuration tracks sandbox creation duration
	SandboxCreateDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "sandbox_create_duration_seconds",
			Help:    "Duration of sandbox creation operations",
			Buckets: prometheus.DefBuckets,
		},
	)

	// SandboxActiveCount tracks active sandboxes
	SandboxActiveCount = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "sandbox_active_count",
			Help: "Number of active sandboxes",
		},
	)

	// RedisOpDuration tracks Redis operation duration
	RedisOpDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "redis_operation_duration_seconds",
			Help:    "Duration of Redis operations",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"operation"},
	)
)
