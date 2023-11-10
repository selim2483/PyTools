import os
from typing import Iterable, List, Optional, Union

from matplotlib import pyplot as plt
import numpy as np
import yaml

_group_type = Union[str, dict[str, Iterable[str]]]
_infos_type = Union[str, dict[str, Iterable[str, str, int]]]
_table_factors_type = Union[str, dict[str, int]]

class LoggingTools :

    def __init__(
            self, 
            groups        :_group_type, 
            infos         :_infos_type, 
            table_factors :_table_factors_type
    ):
        self.groups = self._init_info_dict(groups)
        self.infos = self._init_info_dict(infos)
        self.table_factors = self._init_info_dict(table_factors)

    def _init_info_dict(self, cfg:Union[str, dict, None]) -> dict:
        if isinstance(cfg, str):
            with open(cfg, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)
            return cfg
        elif isinstance(cfg, dict):
            return cfg
        else:
            raise ValueError("'cfg' argument should be of type 'str' or dict")

    @classmethod
    def from_path(cfg:str):
        
    def plot_graph(
            x:np.ndarray, 
            mean_dict:dict[str, float], 
            group_name:str, 
            names:List[str],
            logdir:str,
            group_labels:bool=False,
            type:str="bar",
            **kwargs) :
        
        group = GROUPS[group_name]
        r = 0.6
        total_width = kwargs.get("total_width", 0.8)

        fig = plt.figure(figsize=(r * (3*len(names) + 4) * total_width, r * 6))
        axs = [fig.add_subplot(111)]
        axs[0].set_xlabel("Model")
        axs[0].set_title(f"Comparison {group_name} metrics")
        axs[0].set_xlim(- 0.5, (len(names) - 0.5))

        width = total_width / len(group)
        graphs = []
        for i, metric in enumerate(group) :
            if metric not in mean_dict.keys() :
                pass

            color, label, maxi = INFOS[metric]

            # Scaling
            if i!=0 and group_labels :
                new_max = max(
                    axs[0].get_ylim()[1], 
                    maxi, 
                    *[1.05 * m  for m in mean_dict[metric]])
            else :
                new_max = max(maxi, *[1.05 * m  for m in mean_dict[metric]])
            
            # Add new axes if needed
            if i!=0 and not group_labels :
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
            if i>=2 and not group_labels :
                axs[i].spines['right'].set_position(('outward', 80 * (i - 1))) 
                axs[i].set_ylabel(label)

        # Complete legend and layout
        axs[0].legend(handles=graphs, loc='best')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(names)
        plt.tight_layout()

        plt.savefig(os.path.join(logdir, f"{group_name}.png"))

    def write_latax_table(
            mean_dict:dict[str, float], 
            group_name:str, 
            names:List[str],
            logdir:str,
            init=False) :
        
        def refactor(metric, value) :
            if group_name=="histogram" :
                if "band" in metric :
                    return str(round(10**6 * value, 3))
                else :
                    return str(round(10**6 * value, 1))
            elif "spectral" in metric :
                return str(round(10**3 * value, 2))
            elif "hist" in metric :
                return str(round(10**3 * value, 3))
            elif "SIFID" in metric :
                return str(round(10**-3 * value, 2))
            else :
                return str(round(value, 2))
            
        def scale_str(metric) :
            if group_name=="histogram" and "stochastic" in metric :
                return " $(\cdot10^{{-6}})$"
            elif "spectral" in metric or "hist" in metric :
                return " $(\cdot10^{{-3}})$"
            elif "SIFID" in metric :
                return " $(\cdot10^{{3}})$"
            else :
                return ""
        
        group = GROUPS[group_name]
        m, n = len(group), len(names)
        titles = [INFOS[metric][1] + scale_str(metric) for metric in group]
        table = f'''
    {group_name}
    \\begin{{tabular}}{{|l|{'c|' * m}}}
        \\hline
        & {' & '.join(titles)} \\\\ \\hline\n'''

        for i, name in enumerate(names) :
            metrics = [refactor(metric, mean_dict[metric][i]) for metric in group]
            table += f"    {name} & {' & '.join(metrics)} \\\\ \\hline\n"
        table += "\\end{tabular}\n\n" 

        with open(os.path.join(logdir, "tables.tex"), "w" if init else "a") as f :
            f.write(table)
        
class 