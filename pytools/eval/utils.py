import os
from typing import Callable, Iterable, List, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import yaml


_group_type = Union[str, dict[str, Iterable[str]]]
_infos_type = Union[str, dict[str, Iterable[str, str, int]]]

class LoggingTools :
    """Bunch of usefull methods for logging inference results, metrics or
    images.

    Args:
        groups (_group_type): dict of the considered groups of metrics.
        infos (_infos_type): dict of the informations regarding the considered
            metrics : color, name, scaling for histograms.
    """
    def __init__(self, groups:_group_type, infos:_infos_type):
        self.groups = self._init_info_dict(groups)
        self.infos = self._init_info_dict(infos)

    @classmethod
    def _init_info_dict(cls, cfg:Union[str, dict, None]) -> dict:
        """Instantiate informations dicts depending of the type of input : if 
        is a dict, the input is returned as if in output, if it is a path
        (i.e. string input), the method fetch for the yaml file containing the
        wanted information dict.

        Args:
            cfg (Union[str, dict, None]): dict or yaml file path where to
                fetch the dict

        Raises:
            ValueError: ``cfg`` argument should be of type 'str' or dict.

        Returns:
            dict: info dict.
        """
        if isinstance(cfg, str):
            with open(cfg, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)
            return cfg
        elif isinstance(cfg, dict):
            return cfg
        else:
            raise ValueError(
                "'cfg' argument should be of type 'str' or dict.")

    @classmethod
    def from_path(cls, cfg:str) :
        """Class method to construct LoggingTools object from a yaml file
        containing the necessary dicts. 

        Args:
            cfg (str): yaml file path.

        Returns:
            LoggingTools: LoggingTools object.
        """
        return cls(**cls._init_info_dict(cfg))
    
    def plot_comparison_graph(
            self,
            mean_dict:dict[str, float], 
            group_name:str, 
            names:List[str],
            logdir:str,
            group_scales:bool=False,
            type:str="bar",
            **kwargs) : 
        """Plots a comparison graph between different tested models for a
        given group of metrics. The graph can be an histogram or a line graph,
        using the ``type`` argument. 

        Args:
            mean_dict (dict[str, float]): dict containing the lists of the
                mean metrics values for each model.
            group_name (str): name of the metric group.
            names (List[str]): models names
            logdir (str): directory where to log graph.
            group_scales (bool, optional): Whether to group scales or not. 
                Defaults to False.
            type (str, optional): type of wanted graph : 'bar' or 'line'. 
                Defaults to 'bar'.
        """    
        group = self.groups[group_name]
        r = 0.6
        total_width = kwargs.get("total_width", 0.8)

        fig = plt.figure(figsize=(r * (3*len(names) + 4) * total_width, r * 6))
        axs = [fig.add_subplot(111)]
        axs[0].set_xlabel("Model")
        axs[0].set_title(f"Comparison {group_name} metrics")
        axs[0].set_xlim(- 0.5, (len(names) - 0.5))

        width = total_width / len(group)
        graphs = []
        x = np.arange(len(names))
        for i, metric in enumerate(group) :
            if metric not in mean_dict.keys() :
                pass

            color, label, maxi = self.infos[metric]

            # Scaling
            if i!=0 and group_scales :
                new_max = max(
                    axs[0].get_ylim()[1], 
                    maxi, 
                    *[1.05 * m  for m in mean_dict[metric]])
            else :
                new_max = max(maxi, *[1.05 * m  for m in mean_dict[metric]])
            
            # Add new axes if needed
            if i!=0 and not group_scales :
                axs.append(axs[0].twinx())
                
            axs[-1].set_ylim(0, new_max)
            axs[-1].set_ylabel(label)
            if type=="bar" :
                graphs.append(
                    axs[-1].bar(
                        x - total_width / 2 + (i + 0.5) * width,
                        mean_dict[metric],
                        width=width,
                        color=color,
                        label=label
                    )
                )
            elif type=="line" :
                graphs.append(
                    axs[-1].plot(
                        x,
                        mean_dict[metric],
                        color=color,
                        label=label
                    )
                )
            
            # Add y axis if needed
            if i>=2 and not group_scales :
                axs[i].spines['right'].set_position(('outward', 80 * (i - 1))) 
                axs[i].set_ylabel(label)

        # Complete legend and layout
        axs[0].legend(handles=graphs, loc='best')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(names)
        plt.tight_layout()

        plt.savefig(os.path.join(logdir, f"{group_name}.png"))

    def write_latex_table(
            self,
            mean_dict:dict[str, float], 
            group_name:str, 
            names:List[str],
            logdir:str,
            refactor_fn:Callable[[str, str, float], str]=lambda x: str(x),
            scale_title_fn:Callable[[str, str], str]=lambda x: "",
            init=False) :
        """Write and save a LaTeX formated table with the provided metrics.

        Args:
            mean_dict (dict[str, float]): dict containing the lists of the
                mean metrics values for each model.
            group_name (str): name of the metric group.
            names (List[str]): models names
            logdir (str): directory where to log graph.
            refactor_fn (Callable[[str, str, float], str]): refactor function
                to use for metrics scaling and rounding.
                Defaults to lambda x: str(x).
            scale_title_fn (_type_, optional): function to use to add the
                scaling factor in the table titles.
                Defaults to lambda x: "".
            init (bool, optional): Flag to declare that the table file needs
                to be initialized. The preceding file will then be overwrited.
                Defaults to ``False``.
        """
        group = self.groups[group_name]
        m, n = len(group), len(names)
        titles = [
            self.infos[metric][1] 
            + scale_title_fn(group_name, metric) 
            for metric in group
        ]

        table = f'''
    {group_name}
    \\begin{{tabular}}{{|l|{'c|' * m}}}
        \\hline
        & {' & '.join(titles)} \\\\ \\hline\n'''

        for i, name in enumerate(names) :
            metrics = [
                refactor_fn(group_name, metric, mean_dict[metric][i]) 
                for metric in group
            ]
            table += f"    {name} & {' & '.join(metrics)} \\\\ \\hline\n"
        table += "\\end{tabular}\n\n" 

        table_path = os.path.join(logdir, "tables.tex")
        with open(table_path, "w" if init else "a") as f :
            f.write(table)

