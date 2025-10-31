// helpers.go

package main

import (
	"math"
)

// floatEquals checks if two floats are equal within a tolerance.
// It takes in two float a, b and a tolerance tol. It returns true if the absolute difference between a and b is less than or equal to tol.
func floatEquals(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
