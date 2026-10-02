"""Synthetic sample pools written by similar_samples_llms_eval_lite/run_synthesis.sh
(keep the two in sync): pool name -> (directory under $SIMIT_DATA, candidates per query)."""
POOLS = {
    "v2": ("generated_samples_discard_unverified_v2", 6),
    "nodf4": ("generated_samples_discard_unverified_no_diff_filtering_4_samples", 4),
    "no_diversity": ("generated_samples_no_diversity_prompt_no_diff_filtering_4_samples", 4),
    "force_natural": ("generated_samples_force_natural_no_diff_filtering_4_samples", 4),
    "keep_unverified": ("generated_samples_keep_unverified_no_diff_filtering_4_samples", 4),
    "fixed_question": ("generated_samples_fixed_question_prompt_no_diff_filtering_4_samples", 4),
    "fixed_image": ("generated_samples_fixed_image_prompt_no_diff_filtering_4_samples", 4),
}
