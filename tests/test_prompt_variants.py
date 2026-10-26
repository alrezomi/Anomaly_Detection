import unittest
from unittest.mock import patch

from pipeline_settings import resolve_pipeline_config
from rynnbrain_vlm import prompts


class PromptVariantTests(unittest.TestCase):
    def test_default_a_preserves_all_existing_prompt_forms(self):
        for temporal in (False, True):
            self.assertEqual(prompts.task_context_prompt('pick', temporal=temporal),
                             prompts.task_context_prompt_a('pick', temporal=temporal))
            for mode in ('raw', 'heatmap', 'raw_heatmap'):
                self.assertEqual(prompts.evaluation_prompt_multiturn('pick', mode, temporal=temporal),
                                 prompts.evaluation_prompt_multiturn_a('pick', mode, temporal=temporal))

    def test_b_dispatches_both_turns_and_rejects_unknown_variant(self):
        with patch.object(prompts, 'task_context_prompt_b', return_value='nominal B') as nominal, \
             patch.object(prompts, 'evaluation_prompt_multiturn_b', return_value='test B') as evaluation:
            self.assertEqual(prompts.task_context_prompt('pick', variant='B'), 'nominal B')
            self.assertEqual(prompts.evaluation_prompt_multiturn('pick', 'raw', variant='B'), 'test B')
            nominal.assert_called_once_with('pick', temporal=False)
            evaluation.assert_called_once_with('pick', 'raw', temporal=False)
        with self.assertRaises(ValueError):
            prompts.task_context_prompt('pick', variant='C')

    def test_b_paths_are_separate_repeatable_and_leave_loaded_lora_unchanged(self):
        config = {'output_dir': '/outputs/experiment', 'rynnbrain': {
            'prompt_variant': 'B', 'model': {'lora_adapter_path': '/saved/adapter'},
            'output_dir': '/saved/rynnbrain',
            'lora': {'training': {'output_dir': '/saved/lora_adapter'}},
            'cop_classifier': {'model_paths': {'raw': '/saved/raw_logistic.npz'},
                               'training': {'input_dir': '/saved/vectors', 'output_file': '/saved/raw_logistic.npz'}}}}
        result = resolve_pipeline_config(config)
        vlm = result['rynnbrain']
        self.assertTrue(vlm['output_dir'].endswith('rynnbrain_prompt_b'))
        self.assertTrue(vlm['cop_classifier']['model_paths']['raw'].endswith('raw_logistic_prompt_b.npz'))
        self.assertTrue(vlm['cop_classifier']['training']['input_dir'].endswith('vectors_prompt_b'))
        self.assertTrue(vlm['lora']['training']['output_dir'].endswith('lora_adapter_prompt_b'))
        self.assertEqual(vlm['model']['lora_adapter_path'], '/saved/adapter')
        self.assertEqual(resolve_pipeline_config(result), result)
        self.assertEqual(config['rynnbrain']['output_dir'], '/saved/rynnbrain')
        config['rynnbrain']['prompt_variant'] = 'A'
        self.assertEqual(resolve_pipeline_config(config)['rynnbrain']['output_dir'], '/saved/rynnbrain')

    def test_b_default_paths_are_idempotent(self):
        result = resolve_pipeline_config({'rynnbrain': {'prompt_variant': 'B'}})
        self.assertEqual(resolve_pipeline_config(result), result)


if __name__ == '__main__':
    unittest.main()
