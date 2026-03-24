import tseslint from '@typescript-eslint/eslint-plugin';
import tsparser from '@typescript-eslint/parser';
import prettier from 'eslint-plugin-prettier';
import prettierConfig from 'eslint-config-prettier';

export default [
	{
		files: ['**/*.ts'],
		languageOptions: {
			parser: tsparser,
		},
		plugins: {
			'@typescript-eslint': tseslint,
			prettier,
		},
		rules: {
			...tseslint.configs.recommended.rules,
			...prettierConfig.rules,
			'prettier/prettier': ['error', { endOfLine: 'auto' }],
		},
	},
	{
		ignores: ['node_modules/**', 'dist/**'],
	},
];
