import tempfile
import textwrap

from rlhf_platform.distributed.topology import ModelRole, ParallelStrategy, load_topology_from_yaml


def test_load_topology_from_yaml_loads_model_placements():
    config_text = textwrap.dedent("""
        cluster:
          num_nodes: 2
          gpus_per_node: 1
          total_gpus: 2
        models:
          actor:
            model_name: "opt-1.3b"
            num_gpus: 1
            tp_degree: 1
            dp_degree: 1
            precision: bf16
            devices: [0]
            node_ids: [0]
          critic:
            model_name: "opt-1.3b"
            num_gpus: 1
            tp_degree: 1
            dp_degree: 1
            precision: bf16
            devices: [0]
            node_ids: [0]
          reference:
            model_name: "opt-1.3b"
            num_gpus: 1
            precision: fp32
            devices: [0]
            node_ids: [1]
          reward:
            model_name: "deberta-v3-large"
            num_gpus: 1
            precision: fp32
            devices: [0]
            node_ids: [1]
    """)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_file:
        temp_file.write(config_text)
        temp_file.flush()
        topology = load_topology_from_yaml(temp_file.name, rank=0, local_rank=0)

    assert topology.actor_placement is not None
    assert topology.actor_placement.role == ModelRole.ACTOR
    assert topology.actor_placement.strategy == ParallelStrategy.FSDP_TP
    assert topology.actor_placement.device_list == [0]
    assert topology.reference_placement is not None
    assert topology.reference_placement.strategy.name == "INFERENCE"
    assert topology.reward_placement is not None


def test_topology_validate_rejects_conflicting_assignments():
    config_text = textwrap.dedent("""
        cluster:
          num_nodes: 1
          gpus_per_node: 1
          total_gpus: 1
        models:
          actor:
            model_name: "opt-1.3b"
            num_gpus: 1
            tp_degree: 1
            dp_degree: 1
            precision: bf16
            devices: [0]
            node_ids: [0]
          critic:
            model_name: "opt-1.3b"
            num_gpus: 1
            tp_degree: 1
            dp_degree: 1
            precision: bf16
            devices: [0]
            node_ids: [0]
          reference:
            model_name: "opt-1.3b"
            num_gpus: 1
            precision: fp32
            devices: [0]
            node_ids: [0]
          reward:
            model_name: "deberta-v3-large"
            num_gpus: 1
            precision: fp32
            devices: [0]
            node_ids: [0]
    """)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_file:
        temp_file.write(config_text)
        temp_file.flush()
        try:
            load_topology_from_yaml(temp_file.name, rank=0, local_rank=0)
        except AssertionError as exc:
            assert "assigned to both active training and frozen inference groups" in str(exc)
        else:
            raise AssertionError(
                "load_topology_from_yaml() should raise on overlapping active and inference placements."
            )
