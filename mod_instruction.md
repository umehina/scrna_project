# Go Module Setup Instructions

Follow these steps to enable Go modules and install dependencies for this project.

---

## 1. Enable Go Modules

In your terminal, navigate to the project directory:

```bash
cd scrna_project
```

Then enable Go modules:

```bash
export GO111MODULE="on"
```

---

## 2. Install a New Module or Package

To install a module, run:

```bash
go get -t -u -v $MODPATH
```

Replace `$MODPATH` with the actual module path.  
For example, for GoLearn:

```bash
go get -t -u -v github.com/sjwhitworth/golearn
```

---

## 3. Verify Installation

Open your `go.mod` file and check that the new dependency appears.  
It should look something like this:

```
require github.com/sjwhitworth/golearn v0.0.0-20221228163002-74ae077eafb2 // indirect
```

You can also verify it from the command line:

```bash
go list -m all | grep golearn
```

If nothing is returned, try one of the following:
- Import the module in one of your Go files and run the program, **or**

- Force Go to download it:

```bash
go mod download $MODPATH
```

---

## 4. Example

```bash
# Example: Installing GoLearn
go get -t -u -v github.com/sjwhitworth/golearn
go mod download github.com/sjwhitworth/golearn
```

Afterward, confirm that it appears in `go.mod` and is ready for use.

---

**Tip:**  
If you modify or remove dependencies, always tidy up your module:

```bash
go mod tidy
```

