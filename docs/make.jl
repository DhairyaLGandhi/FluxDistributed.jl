using Documenter
using FluxDistributed

DocMeta.setdocmeta!(FluxDistributed, :DocTestSetup, :(using FluxDistributed); recursive = true)
makedocs(modules = [FluxDistributed],
         doctest = VERSION == v"1.6",
         sitename = "Data Parallel Training",
         pages = ["Home" => "index.md",
                  "Training" => "training.md",
                  "Datasets" => "datasets.md"],
         format = Documenter.HTML(
             analytics = "UA-36890222-9",
             assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"),
         )

deploydocs(repo = "github.com/DhairyaLGandhi/FluxDistributed.jl.git",
           target = "build",
           devbranch = "main",
           push_preview = true)
