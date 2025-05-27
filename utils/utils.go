package utils

import (
	"sync"

	"go.uber.org/zap"
)

var (
	logger *zap.Logger
	once   sync.Once
)

// GetLogger initializes and returns a singleton zap.Logger instance.
func GetLogger() *zap.Logger {
	once.Do(func() {
		config := zap.NewDevelopmentConfig()
		config.Encoding = "console"
		config.DisableStacktrace = true

		// Disable including caller information in logs
		config.EncoderConfig.CallerKey = ""

		var err error
		logger, err = config.Build()
		if err != nil {
			// Fallback to a basic logger if configuration fails
			productionConfig := zap.NewProductionConfig()
			productionConfig.EncoderConfig.CallerKey = ""
			logger, _ = productionConfig.Build()
		}
	})

	return logger
}
