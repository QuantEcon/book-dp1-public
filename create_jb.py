import os
import shutil

julia_src_dir = './source_code_jl/'
python_src_dir = './source_code_py/'
julia_dst_dir = './code-book/jl/'
python_dst_dir = './code-book/py/'

def remove_files(root_src_dir, root_dst_dir):
    for file in os.listdir(root_src_dir):
        if os.path.isfile(os.path.join(root_src_dir, file)):
            os.remove(os.path.join(root_dst_dir, file))
        else:
            shutil.rmtree(os.path.join(root_dst_dir, file))

#################################### Julia Version #################################################

shutil.copytree(julia_src_dir, julia_dst_dir, dirs_exist_ok=True)

header = """---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.9
---

(placeholder_text)=
```{raw} jupyter
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```
"""

contents = """
```{contents} Contents
:depth: 2
```
"""
chapter_meta = {
    "introductions": {
        "name": "Chapter 1: Introductions",
        "subs": { # order determines order in chapter
            "two_period_job_search.jl": ["fig_v1(savefig=true)"],
            "compute_spec_rad.jl": [],
            "power_series.jl": [],
            "s_approx.jl": [],
            "linear_iter.jl": [],
            "linear_iter_fig.jl": [], 
            "iid_job_search.jl": ["fig_vseq(savefig=true)", "fig_vstar(savefig=true)"]  # plotting functions are added here to keep source code clean
        }
    },
    "operators_fixed_points": {
        "name": "Chapter 2: Operators and Fixed Points",
        "subs": {
            "lake.jl": ["plot_paths(savefig=true)", "plot_growth(savefig=true)"],
            # "lake_1.jl"
        }
    },
    "markov_dynamics": {
        "name": "Chapter 3: Markov Dynamics",
        "subs": {
            "inventory_sim.jl": ["model = create_inventory_model()", "plot_ts(model; savefig=true)", "plot_hist(model; savefig=true)"],
            "is_irreducible.jl": [],
            #                    "inventory_sim.jl",
            "laborer_sim.jl": [],
            "markov_js.jl": ["plot_main(savefig=true)"],
            "markov_js_with_sep.jl": ["plot_main(savefig=true)", "plot_w_stars(savefig=true)"]
        }
    },
    "optimal_stopping": {
        "name": "Chapter 4: Optimal Stopping",
        "subs": {
            "firm_exit.jl": ["plot_val(savefig=true)", "plot_comparison(savefig=true)"],
            "american_option.jl": ["plot_contours(savefig=true)", "plot_strike(savefig=true)"]
        }
    },
    "markov_decision_processes": {
        "name": "Chapter 5: Markov Decision Processes",
        "subs": {
            "inventory_dp.jl": ["plot_vstar_and_opt_policy(savefig=true)", "plot_ts(savefig=true)"],
            "finite_opt_saving_0.jl": [],
            "finite_opt_saving_1.jl": [],
            "finite_opt_saving_2.jl": ["plot_timing(savefig=true)", "plot_policy()", "plot_time_series(savefig=true)", "plot_histogram(savefig=true)", "plot_lorenz(savefig=true)"],
            "finite_lq.jl": ["plot_policy()", "plot_sim(savefig=true)", "plot_timing(savefig=true)"],
            "firm_hiring.jl": ["plot_policy()", "plot_sim(savefig=true)", "plot_growth(savefig=true)"],
            "modified_opt_savings.jl": ["plot_contours(savefig=true)", "plot_policies(savefig=true)", "plot_time_series(savefig=true)", "plot_histogram(savefig=true)", "plot_lorenz(savefig=true)"]
        }
    },
    "stochastic_discounting": {
        "name": "Chapter 6: Stochastic Discounting",
        "subs": {
            "plot_interest_rates.jl": ["plot_rates(df_nominal, savefig=true)", "plot_rates(df_real, savefig=true)"],
            # "plot_interest_rates_real.jl",
            "pd_ratio.jl": ["plot_main(savefig=true)"],
            "inventory_sdd.jl": ["plot_ts(savefig=true)"]
        }
    },
    "nonlinear_valuation": {
        "name": "Chapter 7: Nonlinear Valuation",
        "subs": {
            "rs_utility.jl": ["plot_v(savefig=true)", "plot_multiple_v(savefig=true)"],
            "ez_utility.jl": ["plot_convergence(savefig=true)", "plot_v(savefig=true)", "vary_gamma(savefig=true)", "vary_alpha(savefig=true)"]
        }
    },
    "recursive_decision_processes": {
        "name": "Chapter 8: Recursive Decision Processes",
        "subs": {
            "quantile_function.jl": [],
            "quantile_js.jl": ["plot_main(savefig=true)"]
        }
    }
}


pkg = """```{code-cell} julia
:tags: [\"remove-cell\"]
using Pkg;
Pkg.activate(\"./\");

using PyCall;
pygui(:tk);
```
"""

try:
    for chapter in chapter_meta.keys():
        chapter_name = chapter_meta[chapter]["name"]
        chapter_subs = chapter_meta[chapter]["subs"]

        with open(f"./code-book/jl/{chapter}.md", "w", encoding='utf-8') as b:
            new_header = header.replace("placeholder_text", chapter_name)
            b.write(new_header)
            b.write(f"# {chapter_name}\n\n")
            b.write(contents)
            b.write("\n\n")

            b.write(pkg)

            b.write("\n")

            for sub in chapter_subs.keys():
                with open(f"./code-book/jl/{sub}", "r", encoding='utf-8') as g:

                    b.write(f"## {sub}\n")
                    b.write(f"```{{code-cell}} julia\n")
                    #b.write(":tags: [\"hide-input\"]\n")

                    text = g.readlines()

                    for line in text:
                        if "plt.show()" not in line:
                            b.write(line)

                    b.write(f"\n```\n")

                    if len(chapter_subs[sub]) > 0:
                        for func_call in chapter_subs[sub]:
                            code = f"\n```{{code-cell}} julia\n{func_call}\n```\n"

                            b.write(code)
except Exception as e:
    print("Error in Julia markdown generation: " + str(e))
    remove_files(julia_src_dir, julia_dst_dir)
    print("Failed! - Directory cleaned")
    exit(1)

#####################################################################################################

#################################### Python Version #################################################

shutil.copytree(python_src_dir, python_dst_dir, dirs_exist_ok=True)

header = """---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(placeholder_text)=
```{raw} jupyter
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```
"""

contents = """
```{contents} Contents
:depth: 2
```
"""
chapter_meta = {
    "introductions": {
        "name": "Chapter 1: Introductions",
        "subs": { # order determines order in chapter
            "two_period_job_search.py": ["fig_v1()"],
            "compute_spec_rad.py": [],
            "power_series.py": [],
            "s_approx.py": [],
            "linear_iter.py": [],
            "linear_iter_fig.py": [],
            "iid_job_search.py": ["fig_vseq()", "fig_vstar()"]  # plotting functions are added here to keep source code clean
        }
    },
    "operators_fixed_points": {
        "name": "Chapter 2: Operators and Fixed Points",
        "subs": {
            "lake.py": ["plot_paths()", "plot_growth()"],
            # "lake_1.py"
        }
    },
    "markov_dynamics": {
        "name": "Chapter 3: Markov Dynamics",
        "subs": {
            "inventory_sim.py": ["model = create_inventory_model()", "plot_ts(model)", "plot_hist(model)"],
            "is_irreducible.py": [],
            #                    "inventory_sim.py",
            "laborer_sim.py": [],
            "markov_js.py": ["plot_main()"],
            "markov_js_with_sep.py": ["plot_main()", "plot_w_stars()"]
        }
    },
    "optimal_stopping": {
        "name": "Chapter 4: Optimal Stopping",
        "subs": {
            "firm_exit.py": ["plot_val()", "plot_comparison()"],
            "american_option.py": ["plot_contours()", "plot_strike()"]
        }
    },
    "markov_decision_processes": {
        "name": "Chapter 5: Markov Decision Processes",
        "subs": {
            "inventory_dp.py": ["plot_vstar_and_opt_policy()", "plot_ts()"],
            "finite_opt_saving_0.py": [],
            "finite_opt_saving_1.py": [],
            "finite_opt_saving_2.py": ["plot_timing()", "plot_policy()", "plot_time_series()", "plot_histogram()", "plot_lorenz()"],
            "finite_lq.py": ["plot_policy()", "plot_sim()", "plot_timing()"],
            "firm_hiring.py": ["plot_policy()", "plot_sim()", "plot_growth()"],
            "modified_opt_savings.py": ["plot_contours()", "plot_policies()", "plot_time_series()", "plot_histogram()", "plot_lorenz()"]
        }
    },
    "stochastic_discounting": {
        "name": "Chapter 6: Stochastic Discounting",
        "subs": {
            "plot_interest_rates.py": ["plot_rates(df_nominal)", "plot_rates(df_real)"],
            # "plot_interest_rates_real.py",
            "pd_ratio.py": ["plot_main()"],
            "inventory_sdd.py": ["plot_ts()"]
        }
    },
    "nonlinear_valuation": {
        "name": "Chapter 7: Nonlinear Valuation",
        "subs": {
            "rs_utility.py": ["plot_v()", "plot_multiple_v()"],
            "ez_utility.py": ["plot_convergence()", "plot_v()", "vary_gamma()", "vary_alpha()"]
        }
    },
    "recursive_decision_processes": {
        "name": "Chapter 8: Recursive Decision Processes",
        "subs": {
            "quantile_function.py": [],
            "quantile_js.py": ["plot_main()"]
        }
    }
}

try:
    for chapter in chapter_meta.keys():
        chapter_name = chapter_meta[chapter]["name"]
        chapter_subs = chapter_meta[chapter]["subs"]

        with open(f"./code-book/py/{chapter}.md", "w", encoding='utf-8') as b:
            new_header = header.replace("placeholder_text", chapter_name)
            b.write(new_header)
            b.write(f"# {chapter_name}\n\n")
            b.write(contents)
            b.write("\n\n")

            b.write("\n")

            for sub in chapter_subs.keys():
                with open(f"./code-book/py/{sub}", "r", encoding='utf-8') as g:

                    b.write(f"## {sub}\n")
                    b.write(f"```{{code-cell}} python3\n")
                    #b.write(":tags: [\"hide-input\"]\n")

                    text = g.readlines()

                    for line in text:
                        if "plt.show()" not in line:
                            b.write(line)

                        if "import matplotlib.pyplot as plt" in line:
                            b.write(line + "\nplt.rcParams.update({\"text.usetex\": True, \"font.size\": 14})\n")

                    b.write(f"\n```\n")

                    if len(chapter_subs[sub]) > 0:
                        for func_call in chapter_subs[sub]:
                            code = f"\n```{{code-cell}} python\n{func_call}\n```\n"

                            b.write(code)
except Exception as e:
    print("Error in Python markdown generation: " + str(e))
    remove_files(julia_src_dir, julia_dst_dir)
    remove_files(python_src_dir, python_dst_dir)
    print("Failed! - Directory cleaned")
    exit(1)

#####################################################################################################

remove_files(python_src_dir, python_dst_dir)
remove_files(julia_src_dir, julia_dst_dir)
