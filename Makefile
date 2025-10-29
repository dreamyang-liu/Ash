all:
	cd k8s-scaffold && $(MAKE)
	cd sandbox-recipe && $(MAKE)

clean:
	cd scaffold && $(MAKE) clean

.PHONY: all clean