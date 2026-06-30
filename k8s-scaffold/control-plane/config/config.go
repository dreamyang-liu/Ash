package config

import (
	"log"
	"os"
	"strconv"
)

// Config holds all the environment-based configuration
type Config struct {
	Namespace          string
	WaitDeployReadySec int
	WaitSvcIPSec       int
	RedisHost          string
	RedisPort          int
	RedisDB            int
	ServiceAccountName string
	ReconcileInterval  int
}

// getEnv returns the environment variable value or a default
func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

// getEnvInt returns the environment variable as int or a default
func getEnvInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
		log.Printf("Warning: invalid integer value for %s: %s, using default %d", key, v, defaultVal)
	}
	return defaultVal
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *Config {
	return &Config{
		Namespace:          getEnv("TARGET_NAMESPACE", "ash"),
		WaitDeployReadySec: getEnvInt("WAIT_DEPLOY_READY_SEC", 120),
		WaitSvcIPSec:       getEnvInt("WAIT_SVC_IP_SEC", 120),
		RedisHost:          getEnv("REDIS_HOST", "localhost"),
		RedisPort:          getEnvInt("REDIS_PORT", 6379),
		RedisDB:            getEnvInt("REDIS_DB", 0),
		ServiceAccountName: getEnv("SERVICE_ACCOUNT_NAME", "default"),
		ReconcileInterval:  getEnvInt("RECONCILE_INTERVAL", 30),
	}
}
