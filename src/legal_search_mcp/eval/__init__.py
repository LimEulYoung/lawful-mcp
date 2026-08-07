"""평가 패키지 — lawful 런타임에는 recommended_range 만 남는다.

⚠ recommended_range.py 는 *평가 전용이 아니다*. 프로덕션 도구
  `tools/compute_sentencing_range.py` 가 determine_range / RecommendedRange /
  AppliedFactor / in_range / within_range_position 를 import 해 권고 형량범위를
  계산한다 → **삭제 금지**(모듈명이 eval 이라고 dead code 로 오인하지 말 것).

레퍼런스 하네스의 나머지 eval 모듈(metrics / runner / tier1 양형 / tier2 변시 객관식)은
프로덕션 미사용이라 vendoring 시 제외됨.
"""
